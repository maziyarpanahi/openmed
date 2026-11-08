// Verify-only hashing and Ed25519 for the native (dart:ffi) build.
//
// The arithmetic relies on 64-bit integers, so this library is only compiled
// where `dart:io` and `dart:ffi` exist. It handles public inputs only
// (manifests, signatures, and keys), so constant-time arithmetic is not
// required. It never signs and never holds private keys.

import 'dart:typed_data';

const int _mask32 = 0xffffffff;

const List<int> _sha256K = <int>[
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, //
  0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
  0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
  0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
  0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
  0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
  0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
  0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
  0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
  0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

/// Incremental SHA-256 (FIPS 180-4).
class Sha256 {
  /// Creates an empty hasher.
  Sha256() {
    _state.setAll(0, const <int>[
      0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, //
      0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    ]);
  }

  final Uint32List _state = Uint32List(8);
  final Uint32List _schedule = Uint32List(64);
  final Uint8List _buffer = Uint8List(64);
  int _buffered = 0;
  int _length = 0;
  bool _closed = false;

  /// Adds [data] to the digest.
  void add(List<int> data) {
    if (_closed) {
      throw StateError('the hasher is closed');
    }
    _length += data.length;
    int offset = 0;
    while (offset < data.length) {
      final int take = (64 - _buffered) < (data.length - offset)
          ? 64 - _buffered
          : data.length - offset;
      for (int index = 0; index < take; ++index) {
        _buffer[_buffered + index] = data[offset + index] & 0xff;
      }
      _buffered += take;
      offset += take;
      if (_buffered == 64) {
        _block();
        _buffered = 0;
      }
    }
  }

  /// Finishes the digest and returns its 32 bytes.
  Uint8List close() {
    if (_closed) {
      throw StateError('the hasher is closed');
    }
    final int bits = _length * 8;
    final int padding = _buffered < 56 ? 56 - _buffered : 120 - _buffered;
    final Uint8List tail = Uint8List(padding + 8);
    tail[0] = 0x80;
    for (int index = 0; index < 8; ++index) {
      tail[padding + index] = (bits >> (56 - 8 * index)) & 0xff;
    }
    final int length = _length;
    add(tail);
    _length = length;
    _closed = true;
    final Uint8List digest = Uint8List(32);
    for (int index = 0; index < 8; ++index) {
      digest[index * 4] = _state[index] >> 24;
      digest[index * 4 + 1] = (_state[index] >> 16) & 0xff;
      digest[index * 4 + 2] = (_state[index] >> 8) & 0xff;
      digest[index * 4 + 3] = _state[index] & 0xff;
    }
    return digest;
  }

  static int _rotate(int value, int count) =>
      ((value >> count) | (value << (32 - count))) & _mask32;

  void _block() {
    final Uint32List w = _schedule;
    for (int index = 0; index < 16; ++index) {
      w[index] = (_buffer[index * 4] << 24) |
          (_buffer[index * 4 + 1] << 16) |
          (_buffer[index * 4 + 2] << 8) |
          _buffer[index * 4 + 3];
    }
    for (int index = 16; index < 64; ++index) {
      final int s0 = _rotate(w[index - 15], 7) ^
          _rotate(w[index - 15], 18) ^
          (w[index - 15] >> 3);
      final int s1 = _rotate(w[index - 2], 17) ^
          _rotate(w[index - 2], 19) ^
          (w[index - 2] >> 10);
      w[index] = (w[index - 16] + s0 + w[index - 7] + s1) & _mask32;
    }
    int a = _state[0];
    int b = _state[1];
    int c = _state[2];
    int d = _state[3];
    int e = _state[4];
    int f = _state[5];
    int g = _state[6];
    int h = _state[7];
    for (int index = 0; index < 64; ++index) {
      final int s1 = _rotate(e, 6) ^ _rotate(e, 11) ^ _rotate(e, 25);
      final int choose = (e & f) ^ ((~e & _mask32) & g);
      final int first =
          (h + s1 + choose + _sha256K[index] + w[index]) & _mask32;
      final int s0 = _rotate(a, 2) ^ _rotate(a, 13) ^ _rotate(a, 22);
      final int majority = (a & b) ^ (a & c) ^ (b & c);
      final int second = (s0 + majority) & _mask32;
      h = g;
      g = f;
      f = e;
      e = (d + first) & _mask32;
      d = c;
      c = b;
      b = a;
      a = (first + second) & _mask32;
    }
    _state[0] = (_state[0] + a) & _mask32;
    _state[1] = (_state[1] + b) & _mask32;
    _state[2] = (_state[2] + c) & _mask32;
    _state[3] = (_state[3] + d) & _mask32;
    _state[4] = (_state[4] + e) & _mask32;
    _state[5] = (_state[5] + f) & _mask32;
    _state[6] = (_state[6] + g) & _mask32;
    _state[7] = (_state[7] + h) & _mask32;
  }
}

/// Returns the SHA-256 digest of [data].
Uint8List sha256Bytes(List<int> data) => (Sha256()..add(data)).close();

/// Returns lowercase hex for [bytes].
String hexEncode(List<int> bytes) {
  const String digits = '0123456789abcdef';
  final StringBuffer output = StringBuffer();
  for (final int byte in bytes) {
    output
      ..write(digits[(byte >> 4) & 0x0f])
      ..write(digits[byte & 0x0f]);
  }
  return output.toString();
}

/// Returns `sha256:<hex>` for [data].
String sha256Identity(List<int> data) =>
    'sha256:${hexEncode(sha256Bytes(data))}';

// SHA-512 round constants. Dart's native 64-bit integers wrap on overflow,
// and hex literals above 2^63 denote their two's-complement value.
const List<int> _sha512K = <int>[
  0x428a2f98d728ae22, 0x7137449123ef65cd, 0xb5c0fbcfec4d3b2f, //
  0xe9b5dba58189dbbc, 0x3956c25bf348b538, 0x59f111f1b605d019,
  0x923f82a4af194f9b, 0xab1c5ed5da6d8118, 0xd807aa98a3030242,
  0x12835b0145706fbe, 0x243185be4ee4b28c, 0x550c7dc3d5ffb4e2,
  0x72be5d74f27b896f, 0x80deb1fe3b1696b1, 0x9bdc06a725c71235,
  0xc19bf174cf692694, 0xe49b69c19ef14ad2, 0xefbe4786384f25e3,
  0x0fc19dc68b8cd5b5, 0x240ca1cc77ac9c65, 0x2de92c6f592b0275,
  0x4a7484aa6ea6e483, 0x5cb0a9dcbd41fbd4, 0x76f988da831153b5,
  0x983e5152ee66dfab, 0xa831c66d2db43210, 0xb00327c898fb213f,
  0xbf597fc7beef0ee4, 0xc6e00bf33da88fc2, 0xd5a79147930aa725,
  0x06ca6351e003826f, 0x142929670a0e6e70, 0x27b70a8546d22ffc,
  0x2e1b21385c26c926, 0x4d2c6dfc5ac42aed, 0x53380d139d95b3df,
  0x650a73548baf63de, 0x766a0abb3c77b2a8, 0x81c2c92e47edaee6,
  0x92722c851482353b, 0xa2bfe8a14cf10364, 0xa81a664bbc423001,
  0xc24b8b70d0f89791, 0xc76c51a30654be30, 0xd192e819d6ef5218,
  0xd69906245565a910, 0xf40e35855771202a, 0x106aa07032bbd1b8,
  0x19a4c116b8d2d0c8, 0x1e376c085141ab53, 0x2748774cdf8eeb99,
  0x34b0bcb5e19b48a8, 0x391c0cb3c5c95a63, 0x4ed8aa4ae3418acb,
  0x5b9cca4f7763e373, 0x682e6ff3d6b2b8a3, 0x748f82ee5defb2fc,
  0x78a5636f43172f60, 0x84c87814a1f0ab72, 0x8cc702081a6439ec,
  0x90befffa23631e28, 0xa4506cebde82bde9, 0xbef9a3f7b2c67915,
  0xc67178f2e372532b, 0xca273eceea26619c, 0xd186b8c721c0c207,
  0xeada7dd6cde0eb1e, 0xf57d4f7fee6ed178, 0x06f067aa72176fba,
  0x0a637dc5a2c898a6, 0x113f9804bef90dae, 0x1b710b35131c471b,
  0x28db77f523047d84, 0x32caab7b40c72493, 0x3c9ebe0a15c9bebc,
  0x431d67c49c100d4c, 0x4cc5d4becb3e42b6, 0x597f299cfc657e2a,
  0x5fcb6fab3ad6faec, 0x6c44198c4a475817,
];

int _rotate64(int value, int count) =>
    (value >>> count) | (value << (64 - count));

/// Returns the SHA-512 digest of the concatenation of [parts].
Uint8List sha512Bytes(List<List<int>> parts) {
  int total = 0;
  for (final List<int> part in parts) {
    total += part.length;
  }
  final int padding =
      (total % 128) < 112 ? 112 - (total % 128) : 240 - (total % 128);
  final Uint8List message = Uint8List(total + padding + 16);
  int offset = 0;
  for (final List<int> part in parts) {
    message.setAll(offset, part);
    offset += part.length;
  }
  message[total] = 0x80;
  final int bits = total * 8;
  for (int index = 0; index < 8; ++index) {
    message[message.length - 1 - index] = (bits >>> (8 * index)) & 0xff;
  }
  final List<int> state = <int>[
    0x6a09e667f3bcc908, 0xbb67ae8584caa73b, 0x3c6ef372fe94f82b, //
    0xa54ff53a5f1d36f1, 0x510e527fade682d1, 0x9b05688c2b3e6c1f,
    0x1f83d9abfb41bd6b, 0x5be0cd19137e2179,
  ];
  final List<int> w = List<int>.filled(80, 0);
  for (int block = 0; block < message.length; block += 128) {
    for (int index = 0; index < 16; ++index) {
      int word = 0;
      for (int byte = 0; byte < 8; ++byte) {
        word = (word << 8) | message[block + index * 8 + byte];
      }
      w[index] = word;
    }
    for (int index = 16; index < 80; ++index) {
      final int s0 = _rotate64(w[index - 15], 1) ^
          _rotate64(w[index - 15], 8) ^
          (w[index - 15] >>> 7);
      final int s1 = _rotate64(w[index - 2], 19) ^
          _rotate64(w[index - 2], 61) ^
          (w[index - 2] >>> 6);
      w[index] = w[index - 16] + s0 + w[index - 7] + s1;
    }
    int a = state[0];
    int b = state[1];
    int c = state[2];
    int d = state[3];
    int e = state[4];
    int f = state[5];
    int g = state[6];
    int h = state[7];
    for (int index = 0; index < 80; ++index) {
      final int s1 = _rotate64(e, 14) ^ _rotate64(e, 18) ^ _rotate64(e, 41);
      final int choose = (e & f) ^ (~e & g);
      final int first = h + s1 + choose + _sha512K[index] + w[index];
      final int s0 = _rotate64(a, 28) ^ _rotate64(a, 34) ^ _rotate64(a, 39);
      final int majority = (a & b) ^ (a & c) ^ (b & c);
      final int second = s0 + majority;
      h = g;
      g = f;
      f = e;
      e = d + first;
      d = c;
      c = b;
      b = a;
      a = first + second;
    }
    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
  }
  final Uint8List digest = Uint8List(64);
  for (int index = 0; index < 8; ++index) {
    for (int byte = 0; byte < 8; ++byte) {
      digest[index * 8 + byte] = (state[index] >>> (56 - 8 * byte)) & 0xff;
    }
  }
  return digest;
}

/// Verify-only Ed25519 (RFC 8032 section 5.1.7) built on [BigInt].
///
/// Non-canonical encodings are rejected: `S >= L`, `y >= p`, and `x = 0` with
/// the sign bit set.
class Ed25519Verifier {
  Ed25519Verifier._();

  static final BigInt _p = (BigInt.one << 255) - BigInt.from(19);
  static final BigInt _l = (BigInt.one << 252) +
      BigInt.parse('27742317777372353535851937790883648493');
  static final BigInt _two = BigInt.from(2);
  static final BigInt _d =
      (BigInt.from(-121665) * BigInt.from(121666).modInverse(_p)) % _p;
  static final BigInt _sqrtM1 = _two.modPow((_p - BigInt.one) >> 2, _p);
  static final _Point _identity =
      _Point(BigInt.zero, BigInt.one, BigInt.one, BigInt.zero);
  static final _Point _base = _basePoint();

  static _Point _basePoint() {
    final BigInt y = (BigInt.from(4) * BigInt.from(5).modInverse(_p)) % _p;
    final BigInt x = _recoverX(y, 0)!;
    return _Point(x, y, BigInt.one, (x * y) % _p);
  }

  /// Returns whether [signature] is a valid Ed25519 signature of [message].
  static bool verify(
      List<int> publicKey, List<int> signature, List<int> message) {
    if (publicKey.length != 32 || signature.length != 64) {
      return false;
    }
    final _Point? a = _decompress(publicKey);
    if (a == null) {
      return false;
    }
    final List<int> rBytes = signature.sublist(0, 32);
    final _Point? r = _decompress(rBytes);
    if (r == null) {
      return false;
    }
    final BigInt s = _littleEndian(signature.sublist(32, 64));
    if (s >= _l) {
      return false;
    }
    final Uint8List digest =
        sha512Bytes(<List<int>>[rBytes, publicKey, message]);
    final BigInt h = _littleEndian(digest) % _l;
    return _equal(_multiply(s, _base), _add(r, _multiply(h, a)));
  }

  static BigInt _littleEndian(List<int> bytes) {
    BigInt value = BigInt.zero;
    for (int index = bytes.length - 1; index >= 0; --index) {
      value = (value << 8) | BigInt.from(bytes[index] & 0xff);
    }
    return value;
  }

  static _Point _add(_Point p, _Point q) {
    final BigInt a = ((p.y - p.x) * (q.y - q.x)) % _p;
    final BigInt b = ((p.y + p.x) * (q.y + q.x)) % _p;
    final BigInt c = (_two * p.t * q.t * _d) % _p;
    final BigInt d = (_two * p.z * q.z) % _p;
    final BigInt e = b - a;
    final BigInt f = d - c;
    final BigInt g = d + c;
    final BigInt h = b + a;
    return _Point((e * f) % _p, (g * h) % _p, (f * g) % _p, (e * h) % _p);
  }

  static _Point _multiply(BigInt scalar, _Point point) {
    _Point result = _identity;
    _Point addend = point;
    BigInt remaining = scalar;
    while (remaining > BigInt.zero) {
      if (remaining.isOdd) {
        result = _add(result, addend);
      }
      addend = _add(addend, addend);
      remaining = remaining >> 1;
    }
    return result;
  }

  static bool _equal(_Point p, _Point q) =>
      ((p.x * q.z) - (q.x * p.z)) % _p == BigInt.zero &&
      ((p.y * q.z) - (q.y * p.z)) % _p == BigInt.zero;

  static BigInt? _recoverX(BigInt y, int sign) {
    if (y >= _p) {
      return null;
    }
    final BigInt y2 = y * y;
    final BigInt x2 =
        ((y2 - BigInt.one) * ((_d * y2 + BigInt.one).modInverse(_p))) % _p;
    if (x2 == BigInt.zero) {
      return sign == 0 ? BigInt.zero : null;
    }
    BigInt x = x2.modPow((_p + BigInt.from(3)) >> 3, _p);
    if ((x * x - x2) % _p != BigInt.zero) {
      x = (x * _sqrtM1) % _p;
    }
    if ((x * x - x2) % _p != BigInt.zero) {
      return null;
    }
    if (x.isOdd != (sign == 1)) {
      x = _p - x;
    }
    return x;
  }

  static _Point? _decompress(List<int> encoded) {
    if (encoded.length != 32) {
      return null;
    }
    final BigInt value = _littleEndian(encoded);
    final int sign = ((value >> 255) & BigInt.one).toInt();
    final BigInt y = value & ((BigInt.one << 255) - BigInt.one);
    final BigInt? x = _recoverX(y, sign);
    if (x == null) {
      return null;
    }
    return _Point(x, y, BigInt.one, (x * y) % _p);
  }
}

class _Point {
  const _Point(this.x, this.y, this.z, this.t);

  final BigInt x;
  final BigInt y;
  final BigInt z;
  final BigInt t;
}
