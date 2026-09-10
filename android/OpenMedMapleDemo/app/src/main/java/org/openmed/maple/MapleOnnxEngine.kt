package org.openmed.maple

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer
import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OnnxTensorLike
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import java.io.Closeable
import java.nio.DoubleBuffer
import java.nio.FloatBuffer
import java.nio.ShortBuffer
import kotlin.math.exp
import kotlin.random.Random
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext

data class MapleGenerationRequest(
    val prompt: String,
    val maxNewTokens: Int,
    val temperature: Float,
    val topK: Int = 24,
    val repetitionPenalty: Float = 1.08f,
)

data class MapleGenerationResult(
    val text: String,
    val generatedTokens: Int,
    val elapsedMillis: Long,
) {
    val tokensPerSecond: Double = if (elapsedMillis <= 0L) {
        0.0
    } else {
        generatedTokens * 1000.0 / elapsedMillis
    }
}

internal fun mapleInitialCacheShape(
    inputName: String,
    declaredShape: LongArray,
): LongArray {
    requireBundle(
        declaredShape.size == 4,
        "Cache input $inputName must have rank 4 [batch, heads, past_sequence, head_dim]",
    )
    requireBundle(
        declaredShape[0] == 1L || declaredShape[0] < 0L,
        "Cache input $inputName must support batch size 1",
    )
    requireBundle(
        declaredShape[1] > 0L && declaredShape[3] > 0L,
        "Cache input $inputName must declare static head count and head dimension",
    )
    requireBundle(
        declaredShape[2] <= 0L,
        "Cache input $inputName must allow a zero-length past sequence",
    )
    return longArrayOf(1L, declaredShape[1], 0L, declaredShape[3])
}

class MapleOnnxEngine private constructor(
    private val bundle: InstalledMapleBundle,
    private val environment: OrtEnvironment,
    private val prefillSession: OrtSession,
    private val decodeSession: OrtSession?,
    private val tokenizer: HuggingFaceTokenizer,
) : Closeable {
    @Volatile
    private var closed = false

    suspend fun generate(
        request: MapleGenerationRequest,
        onPartial: suspend (text: String, tokenCount: Int) -> Unit = { _, _ -> },
    ): MapleGenerationResult = withContext(Dispatchers.Default) {
        check(!closed) { "Maple engine is closed" }
        val encoded = tokenizer.encode(request.prompt, false, false).ids
        val limits = bundle.manifest.generation
        require(encoded.isNotEmpty()) { "The tokenizer returned an empty prompt" }
        require(encoded.size <= limits.maxInputTokens) {
            "Input is ${encoded.size} tokens; this mobile bundle allows ${limits.maxInputTokens}"
        }
        val maxNewTokens = request.maxNewTokens.coerceAtMost(
            limits.maxContextTokens - encoded.size,
        )
        require(maxNewTokens > 0) { "The prompt leaves no room for generation" }

        val generated = ArrayList<Long>(maxNewTokens)
        val random = Random(request.prompt.hashCode())
        var retainedResult: OrtSession.Result? = null
        val startNanos = System.nanoTime()
        try {
            for (step in 0 until maxNewTokens) {
                currentCoroutineContext().ensureActive()
                val cachedDecode = step > 0 && decodeSession != null
                val session = if (cachedDecode) decodeSession!! else prefillSession
                val context = LongArray(encoded.size + generated.size) { index ->
                    if (index < encoded.size) encoded[index] else generated[index - encoded.size]
                }
                val stepInputIds = if (cachedDecode) {
                    longArrayOf(context.last())
                } else {
                    context
                }
                val nextResult = runStep(
                    session = session,
                    inputIds = stepInputIds,
                    fullContextLength = context.size,
                    cachedResult = if (cachedDecode) retainedResult else null,
                )
                val nextToken = try {
                    sampleNextToken(
                        result = nextResult,
                        generated = generated,
                        temperature = request.temperature,
                        topK = request.topK,
                        repetitionPenalty = request.repetitionPenalty,
                        random = random,
                    )
                } catch (error: Throwable) {
                    nextResult.close()
                    throw error
                }
                val previousResult = retainedResult
                retainedResult = nextResult
                previousResult?.close()
                if (nextToken in limits.eosTokenIds) {
                    break
                }
                generated += nextToken
                val partial = tokenizer.decode(generated.toLongArray(), true)
                onPartial(partial, generated.size)
            }
        } finally {
            retainedResult?.close()
        }
        val elapsedMillis = (System.nanoTime() - startNanos) / 1_000_000L
        MapleGenerationResult(
            text = tokenizer.decode(generated.toLongArray(), true),
            generatedTokens = generated.size,
            elapsedMillis = elapsedMillis,
        )
    }

    override fun close() {
        if (closed) return
        closed = true
        var failure: Throwable? = null
        listOfNotNull(decodeSession, prefillSession).distinct().forEach { session ->
            try {
                session.close()
            } catch (error: Throwable) {
                if (failure == null) failure = error else failure?.addSuppressed(error)
            }
        }
        try {
            tokenizer.close()
        } catch (error: Throwable) {
            if (failure == null) failure = error else failure?.addSuppressed(error)
        }
        failure?.let { throw it }
    }

    private fun runStep(
        session: OrtSession,
        inputIds: LongArray,
        fullContextLength: Int,
        cachedResult: OrtSession.Result?,
    ): OrtSession.Result {
        val contract = bundle.manifest.graphs
        val created = mutableListOf<OnnxTensor>()
        val inputs = mutableMapOf<String, OnnxTensorLike>()
        fun addLongInput(name: String, values: LongArray) {
            if (name !in session.inputNames) return
            val tensor = OnnxTensor.createTensor(environment, arrayOf(values))
            created += tensor
            inputs[name] = tensor
        }

        try {
            addLongInput(contract.inputIdsName, inputIds)
            addLongInput(contract.attentionMaskName, LongArray(fullContextLength) { 1L })
            addLongInput(
                contract.positionIdsName,
                if (inputIds.size == 1 && fullContextLength > 1) {
                    longArrayOf((fullContextLength - 1).toLong())
                } else {
                    LongArray(inputIds.size) { it.toLong() }
                },
            )

            val cache = bundle.manifest.cache
            val cacheInputNames = cache?.let { cacheContract ->
                session.inputNames.filter { it.startsWith(cacheContract.pastInputPrefix) }
            }.orEmpty()
            if (cachedResult == null) {
                val inputInfo = session.inputInfo
                cacheInputNames.forEach { inputName ->
                    val tensorInfo = inputInfo[inputName]?.info as? TensorInfo
                        ?: throw MapleBundleException(
                            "Cache input is missing tensor metadata: $inputName",
                        )
                    val initialShape = mapleInitialCacheShape(
                        inputName = inputName,
                        declaredShape = tensorInfo.shape,
                    )
                    val tensor = createInitialCacheTensor(initialShape, tensorInfo.type)
                    created += tensor
                    inputs[inputName] = tensor
                }
            } else {
                val cacheContract = cache
                    ?: throw MapleBundleException("Cached decoding needs a cache contract")
                cacheInputNames.forEach { inputName ->
                    val suffix = inputName.removePrefix(cacheContract.pastInputPrefix)
                    val outputName = cacheContract.presentOutputPrefix + suffix
                    val value = cachedResult.get(outputName).orElseThrow {
                        MapleBundleException("Missing cache output: $outputName")
                    }
                    val tensor = value as? OnnxTensorLike
                        ?: throw MapleBundleException("Cache output is not a tensor: $outputName")
                    inputs[inputName] = tensor
                }
            }

            val supported = buildSet {
                add(contract.inputIdsName)
                add(contract.attentionMaskName)
                add(contract.positionIdsName)
                bundle.manifest.cache?.let { cacheContract ->
                    addAll(
                        session.inputNames.filter {
                            it.startsWith(cacheContract.pastInputPrefix)
                        },
                    )
                }
            }
            val unsupported = session.inputNames - supported
            requireBundle(
                unsupported.isEmpty(),
                "Export graph has unsupported required inputs: " +
                    unsupported.sorted().joinToString(),
            )
            return session.run(inputs)
        } finally {
            created.forEach(OnnxTensor::close)
        }
    }

    private fun createInitialCacheTensor(
        shape: LongArray,
        type: OnnxJavaType,
    ): OnnxTensor = when (type) {
        OnnxJavaType.FLOAT -> OnnxTensor.createTensor(
            environment,
            FloatBuffer.allocate(0),
            shape,
        )
        OnnxJavaType.DOUBLE -> OnnxTensor.createTensor(
            environment,
            DoubleBuffer.allocate(0),
            shape,
        )
        OnnxJavaType.FLOAT16,
        OnnxJavaType.BFLOAT16,
        -> OnnxTensor.createTensor(
            environment,
            ShortBuffer.allocate(0),
            shape,
            type,
        )
        else -> throw MapleBundleException(
            "Unsupported cache tensor type: $type",
        )
    }

    private fun sampleNextToken(
        result: OrtSession.Result,
        generated: List<Long>,
        temperature: Float,
        topK: Int,
        repetitionPenalty: Float,
        random: Random,
    ): Long {
        val logitsName = bundle.manifest.graphs.logitsName
        val tensor = result.get(logitsName).orElseThrow {
            MapleBundleException("Missing logits output: $logitsName")
        } as? OnnxTensor ?: throw MapleBundleException("Logits output must be a tensor")
        val info = tensor.info
        val shape = info.shape
        requireBundle(shape.size == 3, "Logits must have shape [batch, sequence, vocabulary]")
        requireBundle(shape[0] == 1L, "Maple mobile inference supports batch size 1")
        val vocabularySize = shape[2].toInt()
        requireBundle(vocabularySize > 0, "Logits vocabulary dimension must be static")
        val buffer = tensor.floatBuffer.duplicate()
        val start = buffer.limit() - vocabularySize
        requireBundle(start >= 0, "Logits tensor is smaller than its vocabulary dimension")
        val candidates = topCandidates(
            buffer = buffer,
            start = start,
            size = vocabularySize,
            topK = topK.coerceIn(1, vocabularySize),
            generated = generated,
            repetitionPenalty = repetitionPenalty,
        )
        if (temperature <= 0.001f || candidates.size == 1) {
            return candidates.first().token.toLong()
        }
        val maxLogit = candidates.maxOf { it.logit }
        val weights = candidates.map { exp(((it.logit - maxLogit) / temperature).toDouble()) }
        val target = random.nextDouble() * weights.sum()
        var cumulative = 0.0
        candidates.forEachIndexed { index, candidate ->
            cumulative += weights[index]
            if (cumulative >= target) return candidate.token.toLong()
        }
        return candidates.last().token.toLong()
    }

    private fun topCandidates(
        buffer: FloatBuffer,
        start: Int,
        size: Int,
        topK: Int,
        generated: List<Long>,
        repetitionPenalty: Float,
    ): List<TokenCandidate> {
        val repeated = generated.takeLast(96).toHashSet()
        val best = ArrayList<TokenCandidate>(topK)
        repeat(size) { token ->
            var logit = buffer.get(start + token)
            if (token.toLong() in repeated && repetitionPenalty > 1f) {
                logit = if (logit >= 0f) logit / repetitionPenalty else logit * repetitionPenalty
            }
            if (best.size < topK || logit > best.last().logit) {
                val insertion = best.indexOfFirst { logit > it.logit }
                    .let { if (it == -1) best.size else it }
                best.add(insertion, TokenCandidate(token, logit))
                if (best.size > topK) best.removeAt(best.lastIndex)
            }
        }
        return best
    }

    private data class TokenCandidate(val token: Int, val logit: Float)

    companion object {
        suspend fun open(bundle: InstalledMapleBundle): MapleOnnxEngine =
            withContext(Dispatchers.IO) {
                val environment = OrtEnvironment.getEnvironment()
                var prefill: OrtSession? = null
                var decode: OrtSession? = null
                var tokenizer: HuggingFaceTokenizer? = null
                try {
                    prefill = createSession(environment, bundle.prefillModel)
                    decode = bundle.decodeModel?.let { decodeModel ->
                        if (decodeModel.canonicalFile == bundle.prefillModel.canonicalFile) {
                            prefill
                        } else {
                            createSession(environment, decodeModel)
                        }
                    }
                    tokenizer = HuggingFaceTokenizer.newInstance(
                        bundle.manifest.resolve(
                            bundle.root,
                            bundle.manifest.tokenizerPath,
                        ).toPath(),
                    )
                    validateSession(prefill, bundle.manifest.graphs)
                    decode?.takeUnless { it === prefill }?.let {
                        validateSession(it, bundle.manifest.graphs)
                    }
                    MapleOnnxEngine(bundle, environment, prefill, decode, tokenizer)
                } catch (error: Throwable) {
                    if (decode !== prefill) decode?.close()
                    prefill?.close()
                    tokenizer?.close()
                    throw error
                }
            }

        private fun createSession(environment: OrtEnvironment, model: java.io.File): OrtSession {
            requireBundle(model.isFile, "Missing ONNX Runtime model: ${model.name}")
            val options = OrtSession.SessionOptions()
            return try {
                options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                options.setIntraOpNumThreads(
                    Runtime.getRuntime().availableProcessors().coerceIn(1, 4),
                )
                options.setInterOpNumThreads(1)
                options.setSessionLogVerbosityLevel(0)
                environment.createSession(model.absolutePath, options)
            } finally {
                options.close()
            }
        }

        private fun validateSession(session: OrtSession, contract: MapleGraphContract) {
            requireBundle(
                contract.inputIdsName in session.inputNames,
                "Export graph is missing ${contract.inputIdsName}",
            )
            requireBundle(
                contract.logitsName in session.outputNames,
                "Export graph is missing ${contract.logitsName}",
            )
        }
    }
}
