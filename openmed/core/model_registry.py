"""Model registry for OpenMed models from HuggingFace collection."""

from __future__ import annotations

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
        elif "44M" in self.model_id:
            return 44
        elif "60M" in self.model_id or "65M" in self.model_id or "66M" in self.model_id:
            return 66
        elif "82M" in self.model_id:
            return 82
        elif "108M" in self.model_id or "109M" in self.model_id or "110M" in self.model_id:
            return 110
        elif "125M" in self.model_id:
            return 125
        elif "135M" in self.model_id:
            return 135
        elif "141M" in self.model_id:
            return 141
        elif "149M" in self.model_id:
            return 149
        elif "166M" in self.model_id:
            return 166
        elif "184M" in self.model_id:
            return 184
        elif "209M" in self.model_id or "210M" in self.model_id or "212M" in self.model_id:
            return 210
        elif "220M" in self.model_id:
            return 220
        elif "278M" in self.model_id or "279M" in self.model_id:
            return 279
        elif "335M" in self.model_id or "340M" in self.model_id:
            return 340
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
        elif "568M" in self.model_id:
            return 568
        elif "600M" in self.model_id:
            return 600
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
        model_id="OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1",
        display_name="PII Detection (SuperClinical Small)",
        category="Privacy",
        specialization="Personally Identifiable Information detection",
        description="Detects PII entities for HIPAA-compliant de-identification including names, emails, phone numbers, addresses, and other protected identifiers",
        entity_types=["first_name", "last_name", "email", "phone_number", "ssn", "date_of_birth", "street_address"],
        size_category="Small",
        recommended_confidence=0.50
    ),

    # OpenMed PII Detection Model Collection (33 models)
    # SuperClinical Family
    "pii_superclinical_large": ModelInfo(
        model_id="OpenMed/OpenMed-PII-SuperClinical-Large-434M-v1",
        display_name="PII Detection - SuperClinical Large",
        category="Privacy",
        specialization="Advanced PII detection for clinical text",
        description="OpenMed's flagship PII detection model with comprehensive HIPAA-compliant de-identification including medical records, SSN, demographics, contact info, and more",
        entity_types=["first_name", "last_name", "email", "phone_number", "ssn", "medical_record_number", "date_of_birth", "street_address", "city", "state", "postcode", "account_number", "api_key", "credit_debit_card", "occupation"],
        size_category="Large",
        recommended_confidence=0.50
    ),
    "pii_superclinical_base": ModelInfo(
        model_id="OpenMed/OpenMed-PII-SuperClinical-Base-184M-v1",
        display_name="PII Detection - SuperClinical Base",
        category="Privacy",
        specialization="Balanced PII detection",
        description="Medium-sized PII detection model balancing speed and accuracy for clinical text de-identification",
        entity_types=["first_name", "last_name", "email", "phone_number", "ssn", "medical_record_number", "date_of_birth", "street_address", "city", "state", "postcode"],
        size_category="Medium",
        recommended_confidence=0.50
    ),
    "pii_superclinical_small": ModelInfo(
        model_id="OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1",
        display_name="PII Detection - SuperClinical Small",
        category="Privacy",
        specialization="Fast PII detection",
        description="Lightweight PII detection model optimized for speed while maintaining good accuracy",
        entity_types=["first_name", "last_name", "email", "phone_number", "ssn", "date_of_birth", "street_address"],
        size_category="Small",
        recommended_confidence=0.50
    ),

    # BioClinicalModern Family
    "pii_bioclinical_modern_large": ModelInfo(
        model_id="OpenMed/OpenMed-PII-BioClinicalModern-Large-395M-v1",
        display_name="PII Detection - BioClinicalModern Large",
        category="Privacy",
        specialization="BioClinical PII detection",
        description="Large BioClinical model for comprehensive PII detection in clinical notes",
        entity_types=["first_name", "last_name", "email", "phone_number", "ssn", "medical_record_number", "date_of_birth", "street_address", "city", "state", "postcode"],
        size_category="Large",
        recommended_confidence=0.50
    ),
    "pii_bioclinical_modern_base": ModelInfo(
        model_id="OpenMed/OpenMed-PII-BioClinicalModern-Base-149M-v1",
        display_name="PII Detection - BioClinicalModern Base",
        category="Privacy",
        specialization="BioClinical PII detection",
        description="Base BioClinical model for balanced PII detection performance",
        entity_types=["first_name", "last_name", "email", "phone_number", "ssn", "medical_record_number", "date_of_birth", "street_address"],
        size_category="Medium",
        recommended_confidence=0.50
    ),

    # BioClinicalBERT
    "pii_bioclinical_bert": ModelInfo(
        model_id="OpenMed/OpenMed-PII-BioClinicalBERT-Base-110M-v1",
        display_name="PII Detection - BioClinicalBERT",
        category="Privacy",
        specialization="BERT-based PII detection",
        description="BioClinicalBERT-based model for reliable PII detection in medical text",
        entity_types=["first_name", "last_name", "email", "phone_number", "ssn", "medical_record_number", "date_of_birth", "street_address"],
        size_category="Medium",
        recommended_confidence=0.50
    ),

    # ClinicDischarge
    "pii_clinic_discharge": ModelInfo(
        model_id="OpenMed/OpenMed-PII-ClinicDischarge-Base-110M-v1",
        display_name="PII Detection - ClinicDischarge",
        category="Privacy",
        specialization="Discharge note PII detection",
        description="Specialized for PII detection in clinical discharge summaries and medical records",
        entity_types=["first_name", "last_name", "email", "phone_number", "ssn", "medical_record_number", "date_of_birth", "street_address"],
        size_category="Medium",
        recommended_confidence=0.50
    ),

    # BiomedBERT Family
    "pii_biomed_bert_large": ModelInfo(
        model_id="OpenMed/OpenMed-PII-BiomedBERT-Large-340M-v1",
        display_name="PII Detection - BiomedBERT Large",
        category="Privacy",
        specialization="Biomedical PII detection",
        description="Large BiomedBERT model for comprehensive biomedical text de-identification",
        entity_types=["first_name", "last_name", "email", "phone_number", "ssn", "medical_record_number", "date_of_birth", "street_address", "city", "state"],
        size_category="Large",
        recommended_confidence=0.50
    ),
    "pii_biomed_bert_base": ModelInfo(
        model_id="OpenMed/OpenMed-PII-BiomedBERT-Base-110M-v1",
        display_name="PII Detection - BiomedBERT Base",
        category="Privacy",
        specialization="Biomedical PII detection",
        description="Base BiomedBERT model for balanced biomedical text de-identification",
        entity_types=["first_name", "last_name", "email", "phone_number", "ssn", "medical_record_number", "date_of_birth", "street_address"],
        size_category="Medium",
        recommended_confidence=0.50
    ),

    # BiomedELECTRA Family
    "pii_biomed_electra_large": ModelInfo(
        model_id="OpenMed/OpenMed-PII-BiomedELECTRA-Large-335M-v1",
        display_name="PII Detection - BiomedELECTRA Large",
        category="Privacy",
        specialization="ELECTRA-based PII detection",
        description="Large ELECTRA model for efficient PII detection in biomedical text",
        entity_types=["first_name", "last_name", "email", "phone_number", "ssn", "medical_record_number", "date_of_birth", "street_address", "city", "state"],
        size_category="Large",
        recommended_confidence=0.50
    ),
    "pii_biomed_electra_base": ModelInfo(
        model_id="OpenMed/OpenMed-PII-BiomedELECTRA-Base-110M-v1",
        display_name="PII Detection - BiomedELECTRA Base",
        category="Privacy",
        specialization="ELECTRA-based PII detection",
        description="Base ELECTRA model for fast and efficient PII detection",
        entity_types=["first_name", "last_name", "email", "phone_number", "ssn", "medical_record_number", "date_of_birth", "street_address"],
        size_category="Medium",
        recommended_confidence=0.50
    ),

    # ClinicalLongformer
    "pii_clinical_longformer": ModelInfo(
        model_id="OpenMed/OpenMed-PII-ClinicalLongformer-Base-149M-v1",
        display_name="PII Detection - ClinicalLongformer",
        category="Privacy",
        specialization="Long document PII detection",
        description="Longformer-based model optimized for long clinical documents and notes",
        entity_types=["first_name", "last_name", "email", "phone_number", "ssn", "medical_record_number", "date_of_birth", "street_address"],
        size_category="Medium",
        recommended_confidence=0.50
    ),

    # ModernMed Family
    "pii_modern_med_large": ModelInfo(
        model_id="OpenMed/OpenMed-PII-ModernMed-Large-395M-v1",
        display_name="PII Detection - ModernMed Large",
        category="Privacy",
        specialization="Modern clinical PII detection",
        description="Large modern architecture for comprehensive clinical PII detection",
        entity_types=["first_name", "last_name", "email", "phone_number", "ssn", "medical_record_number", "date_of_birth", "street_address", "city", "state", "postcode"],
        size_category="Large",
        recommended_confidence=0.50
    ),
    "pii_modern_med_base": ModelInfo(
        model_id="OpenMed/OpenMed-PII-ModernMed-Base-149M-v1",
        display_name="PII Detection - ModernMed Base",
        category="Privacy",
        specialization="Modern clinical PII detection",
        description="Base modern architecture for balanced clinical PII detection",
        entity_types=["first_name", "last_name", "email", "phone_number", "ssn", "medical_record_number", "date_of_birth", "street_address"],
        size_category="Medium",
        recommended_confidence=0.50
    ),

    # QwenMed
    "pii_qwen_med_xlarge": ModelInfo(
        model_id="OpenMed/OpenMed-PII-QwenMed-XLarge-600M-v1",
        display_name="PII Detection - QwenMed XLarge",
        category="Privacy",
        specialization="High-accuracy PII detection",
        description="Extra-large Qwen-based model for maximum PII detection accuracy",
        entity_types=["first_name", "last_name", "email", "phone_number", "ssn", "medical_record_number", "date_of_birth", "street_address", "city", "state", "postcode", "account_number"],
        size_category="XLarge",
        recommended_confidence=0.50
    ),

    # ClinicalBGE Family
    "pii_clinical_bge_large_568m": ModelInfo(
        model_id="OpenMed/OpenMed-PII-ClinicalBGE-Large-568M-v1",
        display_name="PII Detection - ClinicalBGE Large (568M)",
        category="Privacy",
        specialization="BGE-based PII detection",
        description="Large BGE model for high-quality clinical PII detection",
        entity_types=["first_name", "last_name", "email", "phone_number", "ssn", "medical_record_number", "date_of_birth", "street_address", "city", "state", "postcode"],
        size_category="XLarge",
        recommended_confidence=0.50
    ),
    "pii_clinical_bge_large_335m": ModelInfo(
        model_id="OpenMed/OpenMed-PII-ClinicalBGE-Large-335M-v1",
        display_name="PII Detection - ClinicalBGE Large (335M)",
        category="Privacy",
        specialization="BGE-based PII detection",
        description="BGE model for balanced clinical PII detection performance",
        entity_types=["first_name", "last_name", "email", "phone_number", "ssn", "medical_record_number", "date_of_birth", "street_address", "city", "state"],
        size_category="Large",
        recommended_confidence=0.50
    ),

    # EuroMed
    "pii_euro_med": ModelInfo(
        model_id="OpenMed/OpenMed-PII-EuroMed-Large-210M-v1",
        display_name="PII Detection - EuroMed",
        category="Privacy",
        specialization="European clinical PII detection",
        description="Optimized for European clinical text and GDPR compliance",
        entity_types=["first_name", "last_name", "email", "phone_number", "ssn", "medical_record_number", "date_of_birth", "street_address", "city", "state", "postcode"],
        size_category="Medium",
        recommended_confidence=0.50
    ),

    # LiteClinical
    "pii_lite_clinical": ModelInfo(
        model_id="OpenMed/OpenMed-PII-LiteClinical-Small-66M-v1",
        display_name="PII Detection - LiteClinical",
        category="Privacy",
        specialization="Lightweight PII detection",
        description="Ultra-lightweight model for fast PII detection with minimal resource usage",
        entity_types=["first_name", "last_name", "email", "phone_number", "ssn", "date_of_birth", "street_address"],
        size_category="Small",
        recommended_confidence=0.50
    ),

    # mLiteClinical
    "pii_mlite_clinical": ModelInfo(
        model_id="OpenMed/OpenMed-PII-mLiteClinical-Base-135M-v1",
        display_name="PII Detection - mLiteClinical",
        category="Privacy",
        specialization="Multilingual lightweight PII",
        description="Multilingual lightweight model for PII detection across languages",
        entity_types=["first_name", "last_name", "email", "phone_number", "ssn", "medical_record_number", "date_of_birth", "street_address"],
        size_category="Medium",
        recommended_confidence=0.50
    ),

    # FastClinical
    "pii_fast_clinical": ModelInfo(
        model_id="OpenMed/OpenMed-PII-FastClinical-Small-82M-v1",
        display_name="PII Detection - FastClinical",
        category="Privacy",
        specialization="Speed-optimized PII detection",
        description="Optimized for maximum inference speed while maintaining accuracy",
        entity_types=["first_name", "last_name", "email", "phone_number", "ssn", "date_of_birth", "street_address"],
        size_category="Small",
        recommended_confidence=0.50
    ),

    # ClinicalE5 Family
    "pii_clinical_e5_large": ModelInfo(
        model_id="OpenMed/OpenMed-PII-ClinicalE5-Large-335M-v1",
        display_name="PII Detection - ClinicalE5 Large",
        category="Privacy",
        specialization="E5-based PII detection",
        description="Large E5 model for comprehensive clinical PII detection",
        entity_types=["first_name", "last_name", "email", "phone_number", "ssn", "medical_record_number", "date_of_birth", "street_address", "city", "state"],
        size_category="Large",
        recommended_confidence=0.50
    ),
    "pii_clinical_e5_base": ModelInfo(
        model_id="OpenMed/OpenMed-PII-ClinicalE5-Base-109M-v1",
        display_name="PII Detection - ClinicalE5 Base",
        category="Privacy",
        specialization="E5-based PII detection",
        description="Base E5 model for balanced clinical PII detection",
        entity_types=["first_name", "last_name", "email", "phone_number", "ssn", "medical_record_number", "date_of_birth", "street_address"],
        size_category="Medium",
        recommended_confidence=0.50
    ),
    "pii_clinical_e5_small": ModelInfo(
        model_id="OpenMed/OpenMed-PII-ClinicalE5-Small-33M-v1",
        display_name="PII Detection - ClinicalE5 Small",
        category="Privacy",
        specialization="E5-based fast PII detection",
        description="Small E5 model for fast PII detection with minimal resources",
        entity_types=["first_name", "last_name", "email", "phone_number", "ssn", "date_of_birth", "street_address"],
        size_category="Tiny",
        recommended_confidence=0.50
    ),

    # GTEMed
    "pii_gte_med": ModelInfo(
        model_id="OpenMed/OpenMed-PII-GTEMed-Base-149M-v1",
        display_name="PII Detection - GTEMed",
        category="Privacy",
        specialization="GTE-based PII detection",
        description="GTE architecture model for clinical PII detection",
        entity_types=["first_name", "last_name", "email", "phone_number", "ssn", "medical_record_number", "date_of_birth", "street_address"],
        size_category="Medium",
        recommended_confidence=0.50
    ),

    # mSuperClinical
    "pii_msuper_clinical": ModelInfo(
        model_id="OpenMed/OpenMed-PII-mSuperClinical-Large-279M-v1",
        display_name="PII Detection - mSuperClinical",
        category="Privacy",
        specialization="Multilingual advanced PII",
        description="Multilingual SuperClinical model for cross-language PII detection",
        entity_types=["first_name", "last_name", "email", "phone_number", "ssn", "medical_record_number", "date_of_birth", "street_address", "city", "state", "postcode"],
        size_category="Large",
        recommended_confidence=0.50
    ),

    # NomicMed
    "pii_nomic_med": ModelInfo(
        model_id="OpenMed/OpenMed-PII-NomicMed-Large-395M-v1",
        display_name="PII Detection - NomicMed",
        category="Privacy",
        specialization="Nomic-based PII detection",
        description="Nomic architecture model for comprehensive PII detection",
        entity_types=["first_name", "last_name", "email", "phone_number", "ssn", "medical_record_number", "date_of_birth", "street_address", "city", "state", "postcode"],
        size_category="Large",
        recommended_confidence=0.50
    ),

    # mClinicalE5
    "pii_mclinical_e5": ModelInfo(
        model_id="OpenMed/OpenMed-PII-mClinicalE5-Large-560M-v1",
        display_name="PII Detection - mClinicalE5",
        category="Privacy",
        specialization="Multilingual E5 PII detection",
        description="Multilingual E5 model for cross-language clinical PII detection",
        entity_types=["first_name", "last_name", "email", "phone_number", "ssn", "medical_record_number", "date_of_birth", "street_address", "city", "state", "postcode"],
        size_category="XLarge",
        recommended_confidence=0.50
    ),

    # SuperMedical Family
    "pii_super_medical_large": ModelInfo(
        model_id="OpenMed/OpenMed-PII-SuperMedical-Large-355M-v1",
        display_name="PII Detection - SuperMedical Large",
        category="Privacy",
        specialization="Medical-focused PII detection",
        description="Large SuperMedical model for comprehensive medical PII detection",
        entity_types=["first_name", "last_name", "email", "phone_number", "ssn", "medical_record_number", "date_of_birth", "street_address", "city", "state", "postcode"],
        size_category="Large",
        recommended_confidence=0.50
    ),
    "pii_super_medical_base": ModelInfo(
        model_id="OpenMed/OpenMed-PII-SuperMedical-Base-125M-v1",
        display_name="PII Detection - SuperMedical Base",
        category="Privacy",
        specialization="Medical-focused PII detection",
        description="Base SuperMedical model for balanced medical PII detection",
        entity_types=["first_name", "last_name", "email", "phone_number", "ssn", "medical_record_number", "date_of_birth", "street_address"],
        size_category="Medium",
        recommended_confidence=0.50
    ),

    # SnowflakeMed
    "pii_snowflake_med": ModelInfo(
        model_id="OpenMed/OpenMed-PII-SnowflakeMed-Large-568M-v1",
        display_name="PII Detection - SnowflakeMed",
        category="Privacy",
        specialization="Snowflake-based PII detection",
        description="Large Snowflake architecture for high-quality PII detection",
        entity_types=["first_name", "last_name", "email", "phone_number", "ssn", "medical_record_number", "date_of_birth", "street_address", "city", "state", "postcode", "account_number"],
        size_category="XLarge",
        recommended_confidence=0.50
    ),

    # BigMed Family
    "pii_big_med_large_560m": ModelInfo(
        model_id="OpenMed/OpenMed-PII-BigMed-Large-560M-v1",
        display_name="PII Detection - BigMed Large (560M)",
        category="Privacy",
        specialization="Large-scale PII detection",
        description="Extra-large BigMed model for maximum PII detection coverage",
        entity_types=["first_name", "last_name", "email", "phone_number", "ssn", "medical_record_number", "date_of_birth", "street_address", "city", "state", "postcode", "account_number"],
        size_category="XLarge",
        recommended_confidence=0.50
    ),
    "pii_big_med_large_278m": ModelInfo(
        model_id="OpenMed/OpenMed-PII-BigMed-Large-278M-v1",
        display_name="PII Detection - BigMed Large (278M)",
        category="Privacy",
        specialization="Large-scale PII detection",
        description="Large BigMed model for comprehensive PII detection",
        entity_types=["first_name", "last_name", "email", "phone_number", "ssn", "medical_record_number", "date_of_birth", "street_address", "city", "state", "postcode"],
        size_category="Large",
        recommended_confidence=0.50
    ),
}


# ---------------------------------------------------------------------------
# Multilingual PII model generation
# ---------------------------------------------------------------------------

_LANGUAGE_CONFIG = {
    "fr": {"name": "French", "prefix": "French-"},
    "de": {"name": "German", "prefix": "German-"},
    "it": {"name": "Italian", "prefix": "Italian-"},
}

# Keys to skip when generating multilingual variants
# (pii_detection is a legacy alias for pii_superclinical_small)
_PII_SKIP_KEYS = {"pii_detection"}


def _generate_multilingual_pii_models() -> Dict[str, ModelInfo]:
    """Generate registry entries for French, German, and Italian PII models.

    For each of the 33 English PII model templates and each target language,
    creates a new ModelInfo with the language prefix inserted into the
    HuggingFace model ID.

    Returns:
        Dictionary mapping new registry keys to ModelInfo instances.
    """
    generated: Dict[str, ModelInfo] = {}

    for en_key, en_model in list(OPENMED_MODELS.items()):
        if not en_key.startswith("pii_") or en_key in _PII_SKIP_KEYS:
            continue
        if en_model.category != "Privacy":
            continue

        for lang_code, lang_cfg in _LANGUAGE_CONFIG.items():
            lang_name = lang_cfg["name"]
            lang_prefix = lang_cfg["prefix"]

            # Build new registry key: pii_fr_superclinical_large
            new_key = f"pii_{lang_code}_{en_key[4:]}"  # strip "pii_" prefix

            # Insert language prefix into model_id:
            # OpenMed/OpenMed-PII-SuperClinical-Large-434M-v1
            # -> OpenMed/OpenMed-PII-French-SuperClinical-Large-434M-v1
            new_model_id = en_model.model_id.replace(
                "OpenMed/OpenMed-PII-",
                f"OpenMed/OpenMed-PII-{lang_prefix}",
            )

            new_display = en_model.display_name.replace(
                "PII Detection",
                f"PII Detection ({lang_name})",
            )

            new_description = (
                f"{lang_name} language variant: {en_model.description}"
            )

            new_specialization = (
                f"{lang_name} {en_model.specialization}"
            )

            generated[new_key] = ModelInfo(
                model_id=new_model_id,
                display_name=new_display,
                category="Privacy",
                specialization=new_specialization,
                description=new_description,
                entity_types=list(en_model.entity_types),
                size_category=en_model.size_category,
                recommended_confidence=en_model.recommended_confidence,
            )

    return generated


# Merge generated multilingual models into the main registry
OPENMED_MODELS.update(_generate_multilingual_pii_models())


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
    "Privacy": sorted(
        k for k in OPENMED_MODELS if k.startswith("pii_")
    ),
}

# Size-based recommendations
SIZE_RECOMMENDATIONS = {
    "fast": ["disease_detection_tiny", "oncology_detection_tiny", "pii_superclinical_small", "pii_lite_clinical", "pii_fast_clinical"],
    "balanced": ["pharma_detection_supermedical", "genome_detection_bioclinical", "anatomy_detection_electramed", "pii_superclinical_base", "pii_clinical_e5_base"],
    "accurate": ["disease_detection_superclinical", "pharma_detection_superclinical", "oncology_detection_superclinical", "pii_superclinical_large", "pii_qwen_med_xlarge"],
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
        "pii|deidentif|hipaa|phi|protected health|patient name|ssn|medical record|privacy|anonymiz": ("privacy", "Contains PII/de-identification terms"),
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


# ---------------------------------------------------------------------------
# Multilingual PII helpers
# ---------------------------------------------------------------------------

def get_pii_models_by_language(lang: str) -> Dict[str, ModelInfo]:
    """Return all PII models for a given language.

    Args:
        lang: ISO 639-1 language code (en, fr, de, it)

    Returns:
        Dict mapping registry keys to ModelInfo for that language.
    """
    if lang == "en":
        return {
            k: v
            for k, v in OPENMED_MODELS.items()
            if k.startswith("pii_")
            and not any(k.startswith(f"pii_{lc}_") for lc in _LANGUAGE_CONFIG)
        }

    prefix = f"pii_{lang}_"
    return {k: v for k, v in OPENMED_MODELS.items() if k.startswith(prefix)}


def get_default_pii_model(lang: str) -> Optional[str]:
    """Return the default (recommended) PII model_id for a language.

    Args:
        lang: ISO 639-1 language code (en, fr, de, it)

    Returns:
        HuggingFace model ID string, or None if language unsupported.
    """
    from .pii_i18n import DEFAULT_PII_MODELS
    return DEFAULT_PII_MODELS.get(lang)
