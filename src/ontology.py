import re
import json
import os

ONTOLOGY = {
    "Cancer": ["cancer", "malignancy"],
    "Melanoma": ["melanoma"],
    "StomachCancer": ["stomach cancer", "gastric cancer"],
    "CervicalCancer": ["cervical cancer"],
    "Osteosarcoma": ["osteosarcoma", "bone cancer"],

    "Symptom": ["symptom", "symptoms", "sign", "signs"],
    "RiskFactor": ["risk factor", "risk factors"],
    "Biomarker": ["biomarker", "biomarkers"],

    "Diagnosis": ["diagnosis", "diagnostic", "diagnostic test"],
    "Biopsy": ["biopsy", "biopsies"],
    "Imaging": ["imaging", "CT scan", "MRI", "PET scan"],

    "Treatment": ["treatment", "treatments", "therapy"],
    "Surgery": ["surgery", "surgical treatment"],
    "Chemotherapy": ["chemotherapy", "chemo"],
    "RadiationTherapy": ["radiation therapy", "radiotherapy"],
    "Immunotherapy": ["immunotherapy"],
    "TargetedTherapy": ["targeted therapy"],
    "SupportiveCare": ["supportive care"],
    "PalliativeCare": ["palliative care"],

    "CancerStage": ["cancer stage", "cancer staging", "staging"],
    "SideEffect": ["side effect", "side effects", "adverse effect"]
}


RELATIONSHIPS = {
    "Melanoma": ["Cancer"],
    "StomachCancer": ["Cancer"],
    "CervicalCancer": ["Cancer"],
    "Osteosarcoma": ["Cancer"],

    "Cancer": ["Symptom", "RiskFactor", "Biomarker",
               "Diagnosis", "Treatment", "CancerStage"],

    "Diagnosis": ["Biopsy", "Imaging"],
    "Treatment": ["Surgery", "Chemotherapy", "RadiationTherapy",
                  "Immunotherapy", "TargetedTherapy",
                  "SupportiveCare", "PalliativeCare"],
    "Treatment": ["SideEffect"]
}


def normalize(text):
    return re.sub(r"\s+", " ", text.lower()).strip()


def find_concepts(text):
    text = normalize(text)
    concepts = []

    for concept, aliases in ONTOLOGY.items():
        for alias in aliases:
            if alias.lower() in text:
                concepts.append(concept)
                break

    return concepts


def expand_concepts(concepts):
    expanded = set(concepts)

    for concept in concepts:
        if concept in RELATIONSHIPS:
            expanded.update(RELATIONSHIPS[concept])

        for parent, children in RELATIONSHIPS.items():
            if concept in children:
                expanded.add(parent)

    return list(expanded)


def save_ontology():
    os.makedirs("processed_data", exist_ok=True)

    data = {
        "concepts": ONTOLOGY,
        "relationships": RELATIONSHIPS
    }

    with open(
        "processed_data/oncology_ontology.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":

    queries = [
        "How is melanoma diagnosed?",
        "What are the side effects of chemotherapy?",
        "How is stomach cancer treated?",
        "What are the risk factors for cervical cancer?"
    ]

    for query in queries:
        concepts = find_concepts(query)
        print("\nQuery:", query)
        print("Concepts:", concepts)
        print("Expanded:", expand_concepts(concepts))

    save_ontology()
    print("\nOntology saved to processed_data/oncology_ontology.json")