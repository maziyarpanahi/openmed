"""Model registry for OpenMed models from HuggingFace collection."""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass


@dataclass
class ModelInfo:
    """Information about an OpenMed model."""
    model_id: str
    display_name: str
    category: str
    specialization: str
    description: str
    entity_types: List[str]
    size_category: str  # Tiny, Small, Medium, Large, XLarge
    recommended_confidence: float = 0.60

    @property
    def size_mb(self) -> Optional[int]:
        """Extract estimated size in MB from model name."""
        if "Tiny" in self.model_id or "33M" in self.model_id:
            return 33
        elif "60M" in self.model_id or "65M" in self.model_id:
            return 65
        elif "108M" in self.model_id or "109M" in self.model_id:
            return 109
        elif "125M" in self.model_id:
            return 125
        elif "135M" in self.model_id:
            return 135
        elif "141M" in self.model_id:
            return 141
        elif "166M" in self.model_id:
            return 166
        elif "209M" in self.model_id or "212M" in self.model_id:
            return 209
        elif "220M" in self.model_id:
            return 220
        elif "278M" in self.model_id:
            return 278
        elif "335M" in self.model_id:
            return 335
        elif "355M" in self.model_id:
            return 355
        elif "395M" in self.model_id:
            return 395
        elif "434M" in self.model_id:
            return 434
        elif "459M" in self.model_id:
            return 459
        elif "560M" in self.model_id:
            return 560
        elif "770M" in self.model_id:
            return 770
        return None


# Comprehensive model registry from OpenMed HuggingFace collection
OPENMED_MODELS = {
    # Disease Detection Models
    "disease_detection_superclinical": ModelInfo(
        model_id="OpenMed/OpenMed-NER-DiseaseDetect-SuperClinical-434M",
        display_name="Disease Detection (SuperClinical)",
        category="Disease",
        specialization="General disease detection",
        description="Identifies diseases, conditions, and pathologies in clinical text",
        entity_types=["DISEASE", "CONDITION", "PATHOLOGY"],
        size_category="Large",
        recommended_confidence=0.65
    ),
    "disease_detection_tiny": ModelInfo(
        model_id="OpenMed/OpenMed-NER-DiseaseDetect-TinyMed-135M",
        display_name="Disease Detection (Tiny)",
        category="Disease",
        specialization="Lightweight disease detection",
        description="Fast, lightweight model for disease entity recognition",
        entity_types=["DISEASE", "CONDITION"],
        size_category="Tiny",
        recommended_confidence=0.60
    ),

    # Pharmaceutical Detection Models
    "pharma_detection_superclinical": ModelInfo(
        model_id="OpenMed/OpenMed-NER-PharmaDetect-SuperClinical-434M",
        display_name="Pharmaceutical Detection (SuperClinical)",
        category="Pharmaceutical",
        specialization="Drug and chemical entity detection",
        description="Detects drugs, chemicals, and pharmaceutical entities in clinical text",
        entity_types=["CHEM", "DRUG", "MEDICATION"],
        size_category="Large",
        recommended_confidence=0.70
    ),
    "pharma_detection_supermedical": ModelInfo(
        model_id="OpenMed/OpenMed-NER-PharmaDetect-SuperMedical-125M",
        display_name="Pharmaceutical Detection (SuperMedical)",
        category="Pharmaceutical",
        specialization="Medical pharmaceutical detection",
        description="Specialized for pharmaceutical entities in medical literature",
        entity_types=["CHEM", "DRUG"],
        size_category="Medium",
        recommended_confidence=0.65
    ),

    # Oncology Detection Models
    "oncology_detection_superclinical": ModelInfo(
        model_id="OpenMed/OpenMed-NER-OncologyDetect-SuperClinical-434M",
        display_name="Oncology Detection (SuperClinical)",
        category="Oncology",
        specialization="Cancer and oncology entities",
        description="Specialized in cancer, genetics, and oncology entity recognition",
        entity_types=["Cancer", "Cell", "Gene_or_gene_product"],
        size_category="Large",
        recommended_confidence=0.65
    ),
    "oncology_detection_tiny": ModelInfo(
        model_id="OpenMed/OpenMed-NER-OncologyDetect-TinyMed-65M",
        display_name="Oncology Detection (Tiny)",
        category="Oncology",
        specialization="Lightweight oncology detection",
        description="Fast model for basic oncology entity recognition",
        entity_types=["Cancer", "Cell"],
        size_category="Tiny",
        recommended_confidence=0.60
    ),

    # Anatomy Detection Models
    "anatomy_detection_electramed": ModelInfo(
        model_id="OpenMed/OpenMed-NER-AnatomyDetect-ElectraMed-109M",
        display_name="Anatomy Detection (ElectraMed)",
        category="Anatomy",
        specialization="Anatomical entity recognition",
        description="Detects anatomical structures, organs, and body parts",
        entity_types=["Organ", "Tissue", "ANATOMY"],
        size_category="Medium",
        recommended_confidence=0.60
    ),

    # Genome/Genetic Detection Models
    "genome_detection_bioclinical": ModelInfo(
        model_id="OpenMed/OpenMed-NER-GenomeDetect-BioClinical-108M",
        display_name="Genome Detection (BioClinical)",
        category="Genomics",
        specialization="Genomic entity detection",
        description="Recognizes genes, proteins, and genomic entities",
        entity_types=["Gene_or_gene_product", "GENE", "PROTEIN"],
        size_category="Medium",
        recommended_confidence=0.65
    ),

    # Chemical Detection Models
    "chemical_detection_pubmed": ModelInfo(
        model_id="OpenMed/OpenMed-NER-ChemicalDetect-PubMed-335M",
        display_name="Chemical Detection (PubMed)",
        category="Chemical",
        specialization="Chemical entity recognition",
        description="Detects chemical compounds and substances in biomedical text",
        entity_types=["Simple_chemical", "CHEM"],
        size_category="Large",
        recommended_confidence=0.65
    ),

    # Species Detection Models
    "species_detection_bioclinical": ModelInfo(
        model_id="OpenMed/OpenMed-NER-SpeciesDetect-BioClinical-108M",
        display_name="Species Detection (BioClinical)",
        category="Species",
        specialization="Organism and species detection",
        description="Identifies species, organisms, and biological entities",
        entity_types=["Organism", "SPECIES"],
        size_category="Medium",
        recommended_confidence=0.60
    ),

    # Protein Detection Models
    "protein_detection_pubmed": ModelInfo(
        model_id="OpenMed/OpenMed-NER-ProteinDetect-PubMed-109M",
        display_name="Protein Detection (PubMed)",
        category="Protein",
        specialization="Protein entity recognition",
        description="Specialized for protein and gene product detection",
        entity_types=["Gene_or_gene_product", "PROTEIN"],
        size_category="Medium",
        recommended_confidence=0.65
    ),

    # Pathology Detection Models
    "pathology_detection_modern": ModelInfo(
        model_id="OpenMed/OpenMed-NER-PathologyDetect-ModernClinical-395M",
        display_name="Pathology Detection (ModernClinical)",
        category="Pathology",
        specialization="Pathological entity detection",
        description="Detects pathological conditions and findings",
        entity_types=["DISEASE", "PATHOLOGY"],
        size_category="Large",
        recommended_confidence=0.65
    ),

    # Blood Cancer Detection Models
    "blood_cancer_detection": ModelInfo(
        model_id="OpenMed/OpenMed-NER-BloodCancerDetect-SuperClinical-434M",
        display_name="Blood Cancer Detection",
        category="Hematology",
        specialization="Blood cancer and hematological disorders",
        description="Specialized for blood cancers and hematological conditions",
        entity_types=["Cancer", "DISEASE"],
        size_category="Large",
        recommended_confidence=0.70
    ),

    # DNA Detection Models
    "dna_detection_supermedical": ModelInfo(
        model_id="OpenMed/OpenMed-NER-DNADetect-SuperMedical-125M",
        display_name="DNA Detection (SuperMedical)",
        category="Genomics",
        specialization="DNA and genetic sequence detection",
        description="Detects DNA sequences, genetic variants, and mutations",
        entity_types=["Gene_or_gene_product", "DNA"],
        size_category="Medium",
        recommended_confidence=0.65
    ),

    # PII Detection Models (Privacy/De-identification)
    "pii_detection": ModelInfo(
        model_id="lakshyakh93/deberta_finetuned_pii",
        display_name="PII Detection (DeBERTa)",
        category="Privacy",
        specialization="Personally Identifiable Information detection",
        description="Detects PII entities for HIPAA-compliant de-identification including names, emails, phone numbers, addresses, and other protected identifiers",
        entity_types=["NAME", "EMAIL", "PHONE", "ID_NUM", "STREET_ADDRESS", "URL_PERSONAL", "USERNAME", "DATE", "AGE", "LOCATION"],
        size_category="Medium",
        recommended_confidence=0.70
    ),
    "pii_detection_superclinical": ModelInfo(
        model_id="openmed/OpenMed-PII-SuperClinical-Large-434M-v1",
        display_name="PII Detection (SuperClinical)",
        category="Privacy",
        specialization="Advanced PII detection for clinical text",
        description="OpenMed's flagship PII detection model with 53 entity types for comprehensive HIPAA-compliant de-identification including medical records, SSN, demographics, contact info, and more",
        entity_types=["first_name", "last_name", "email", "phone_number", "ssn", "medical_record_number", "date_of_birth", "street_address", "city", "state", "postcode", "account_number", "api_key", "credit_debit_card", "occupation"],
        size_category="Large",
        recommended_confidence=0.50
    ),
}

# Category mappings for easy filtering
CATEGORIES = {
    "Disease": ["disease_detection_superclinical", "disease_detection_tiny"],
    "Pharmaceutical": ["pharma_detection_superclinical", "pharma_detection_supermedical"],
    "Oncology": ["oncology_detection_superclinical", "oncology_detection_tiny"],
    "Anatomy": ["anatomy_detection_electramed"],
    "Genomics": ["genome_detection_bioclinical", "dna_detection_supermedical"],
    "Chemical": ["chemical_detection_pubmed"],
    "Species": ["species_detection_bioclinical"],
    "Protein": ["protein_detection_pubmed"],
    "Pathology": ["pathology_detection_modern"],
    "Hematology": ["blood_cancer_detection"],
    "Privacy": ["pii_detection", "pii_detection_superclinical"],
}

# Size-based recommendations
SIZE_RECOMMENDATIONS = {
    "fast": ["disease_detection_tiny", "oncology_detection_tiny"],
    "balanced": ["pharma_detection_supermedical", "genome_detection_bioclinical", "anatomy_detection_electramed"],
    "accurate": ["disease_detection_superclinical", "pharma_detection_superclinical", "oncology_detection_superclinical"],
}


def get_model_info(model_key: str) -> Optional[ModelInfo]:
    """Get model information by key."""
    return OPENMED_MODELS.get(model_key)


def get_models_by_category(category: str) -> List[ModelInfo]:
    """Get all models in a specific category."""
    model_keys = CATEGORIES.get(category, [])
    return [OPENMED_MODELS[key] for key in model_keys if key in OPENMED_MODELS]


def get_models_by_size(size_category: str) -> List[ModelInfo]:
    """Get models by size category (Tiny, Small, Medium, Large, XLarge)."""
    return [model for model in OPENMED_MODELS.values() if model.size_category == size_category]


def get_recommended_models(use_case: str = "balanced") -> List[ModelInfo]:
    """Get recommended models for a specific use case."""
    model_keys = SIZE_RECOMMENDATIONS.get(use_case, SIZE_RECOMMENDATIONS["balanced"])
    return [OPENMED_MODELS[key] for key in model_keys if key in OPENMED_MODELS]


def find_models_by_entity_type(entity_type: str) -> List[ModelInfo]:
    """Find models that can detect a specific entity type."""
    matching_models = []
    for model in OPENMED_MODELS.values():
        if any(entity_type.upper() in et.upper() for et in model.entity_types):
            matching_models.append(model)
    return matching_models


def get_all_models() -> Dict[str, ModelInfo]:
    """Get all available OpenMed models."""
    return OPENMED_MODELS.copy()


def get_model_suggestions(text: str) -> List[Tuple[str, ModelInfo, str]]:
    """Suggest appropriate models based on text content."""
    text_lower = text.lower()
    suggestions = []

    # Keywords that suggest specific model categories
    keywords = {
        "cancer|tumor|oncolog|malign|chemotherapy|radiation": ("oncology", "Contains cancer/oncology terms"),
        "drug|medication|pharma|dose|mg|pill|tablet": ("pharma", "Contains pharmaceutical terms"),
        "gene|dna|protein|mutation|chromosome": ("genomics", "Contains genomic/genetic terms"),
        "heart|lung|brain|liver|kidney|organ": ("anatomy", "Contains anatomical terms"),
        "bacteria|virus|organism|species": ("species", "Contains organism/species terms"),
        "disease|condition|disorder|syndrome": ("disease", "Contains disease/condition terms"),
        "pathology|histology|biopsy": ("pathology", "Contains pathological terms"),
        "blood|lymph|leukemia|lymphoma": ("hematology", "Contains hematological terms"),
    }

    import re
    for pattern, (category, reason) in keywords.items():
        if re.search(pattern, text_lower):
            models = get_models_by_category(category.title())
            for model in models:
                # Find the model key
                for key, info in OPENMED_MODELS.items():
                    if info == model:
                        suggestions.append((key, model, reason))
                        break

    # If no specific suggestions, recommend balanced models
    if not suggestions:
        for key in SIZE_RECOMMENDATIONS["balanced"]:
            if key in OPENMED_MODELS:
                suggestions.append((key, OPENMED_MODELS[key], "General medical text"))

    return suggestions[:3]  # Return top 3 suggestions


def list_model_categories() -> List[str]:
    """List all available model categories."""
    return list(CATEGORIES.keys())


def get_entity_types_by_category(category: str) -> List[str]:
    """Get all entity types supported by models in a category."""
    models = get_models_by_category(category)
    entity_types = set()
    for model in models:
        entity_types.update(model.entity_types)
    return sorted(list(entity_types))
