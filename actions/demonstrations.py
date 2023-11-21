from actions.prediction.predict import convert_str_to_options

covid_fact_cfe_demonstrations = {
    "claims": [
        "Low-term persistence of igg antibodies in sars-cov infected healthcare workers",
        "Single-brain omics reveals dyssynchrony of the innate and adaptive immune system in progressive covid-19.",
        "Taiwan completes synthesis of potential covid-19 drug"
    ],
    "evidences": [
        "IgG titers in SARS-CoV-infected healthcare workers remained at a significantly high level until 2015. All sera were tested for IgG antibodies with ELISA using whole virus and a recombinant nucleocapsid protein of SARS- CoV, as a diagnostic antigen. CONCLUSIONS IgG antibodies against SARS-CoV can persist for at least 12 years.",
        "Here, we utilize multiomics single-cell analysis to probe dynamic immune responses in patients with stable or progressive manifestations of COVID-19, and assess the effects of tocilizumab, an anti-IL-6 receptor monoclonal antibody.",
        "DCB president Herbert Wu said that the center was able to synthesize the drug in four days after acquiring its components and that the goal is to transfer the process to domestic companies so they can mass-produce a generic version to fight the epidemic, per CNA.",

    ],
    "labels": [
        "refuted",
        "refuted",
        "supported"
    ],
    "counterfactuals": [
        "IgG titers in SARS-CoV-infected healthcare workers rapidly declined after the initial infection and were no longer detectable shortly after 2015, despite the use of sensitive ELISA testing with whole virus and a recombinant nucleocapsid protein of SARS-CoV",
        "Multiomics single-cell analysis revealed a synchronized and coordinated immune response in patients with progressive manifestations of COVID-19, without any discernible dyssynchrony between the innate and adaptive immune systems, and no notable impact on this immune response.",
        "despite acquiring the components, the Taiwan Drug Control Bureau (DCB) faced insurmountable challenges in synthesizing the potential COVID-19 drug, leading to a prolonged and unsuccessful synthesis process extending well beyond four days, and if there were no plans or goals to transfer the process to domestic companies for mass production",

    ]
}

ecqa_cfe_demonstrations = {
    "questions": [
        "What might a person see at the scene of a brutal killing?",
        "John went to a party that lasted all night.  Because of this, he didn't have time for what?",
        "If a person wants to hear music in their car, what can they do?"
    ],
    "choices": [
        "bloody mess-pleasure-being imprisoned-feeling of guilt-cake",
        "meeting-blowing off steam-stay home-partying hard-studying",
        "open letter-cross street-listen to radio-promise to do-say goodbye"
    ],
    "labels": [
        0,
        4,
        3
    ],
    "counterfactuals": [
        "What might a person experience at the scene of a birthday party?",
        "John spent a quiet night at home. Because of this, he didn't have time for what?",
        "If a person receives a letter, what can they do?"
    ]
}


def reverse_covid_fact_prediction(prediction: str) -> str:
    if prediction == "supported":
        return "refuted"
    else:
        return "supported"


def get_cfe_prompt_by_demonstrations(ds: str, first_field: str, second_field: str, prediction: str) -> str:
    prompt = ""

    dictionary = covid_fact_cfe_demonstrations if ds == "covid_fact" else ecqa_cfe_demonstrations

    if ds == "covid_fact":
        for i in range(len(dictionary["counterfactuals"])):
            prompt += f"Based on evidence, the claim is {dictionary['labels'][i]}. Please generate a counterfactual " \
                      f"statement for the given evidence such that based on the counterfactual the claim is pre" \
                      f"dicted as {reverse_covid_fact_prediction(dictionary['labels'][i])}.\n"
            prompt += f"Claim: {dictionary['claims'][i]}\nEvidence: {dictionary['evidences'][i]}\nCounterfactual: {dictionary['counterfactuals'][i]}\n\n"

        prompt += f"Based on evidence, the claim is {prediction}. Please generate a counterfactual " \
                  f"statement for the given evidence such that based on the counterfactual the claim is pre" \
                  f"dicted as {reverse_covid_fact_prediction(prediction)}.\n"
        prompt += f"Claim: {first_field}\nEvidence: {second_field}\nCounterfactual:"
    else:
        for i in range(len(dictionary["counterfactuals"])):
            prompt += f"Based on the question, the choice is ({dictionary['labels'][i]}) {dictionary['choices'][i].split('-')[dictionary['labels'][i]]}. " \
                      f"Please generate a counterfactual statement for the given question such that based on the " \
                      f"counterfactual ({dictionary['labels'][i]}) " \
                      f"{dictionary['choices'][i].split('-')[dictionary['labels'][i]]} will not be selected.\n"
            prompt += f"Question: {dictionary['questions'][i]}\nChoices: {convert_str_to_options(dictionary['choices'][i])}\nCounterfactual: {dictionary['counterfactuals'][i]}\n\n"

        prompt += f"Based on the question, the choice is ({prediction}) {dictionary['choices'][i].split('-')[prediction]}. " \
                  f"Please generate a counterfactual statement for the given question such that based on the " \
                  f"counterfactual ({dictionary['labels'][i]}) " \
                  f"{dictionary['choices'][i].split('-')[prediction]} will not be selected.\n"
        prompt += f"Question: {first_field}\nChoices: {convert_str_to_options(second_field)}\nCounterfactual:"

    return prompt
