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
        2
    ],
    "counterfactuals": [
        "What might a person experience at the scene of a birthday party?",
        "John spent a quiet night at home. Because of this, he didn't have time for what?",
        "If a person receives a letter, what can they do?"
    ]
}

covid_fact_claim_data_augmentation_demonstrations = {
    "original_text": [
        "Coronavirus dons a new term",
        "Fenofibrate decreased the amount of sulfatide which seems beneficial against covid-19",
        "25-hydroxyvitamin d concentrations are higher in patients with positive pcr for sars-cov-2"
    ],
    "augmented_text": [
        "A fresh term emerges for the coronavirus.",
        "The administration of fenofibrate resulted in a decrease in sulfatide levels, suggesting potential benefits in the fight against COVID-19.",
        "Higher levels of 25-hydroxyvitamin D are observed in individuals who have a positive PCR result for SARS-CoV-2."
    ]
}

covid_fact_evidence_data_augmentation_demonstrations = {
    "original_text": [
        "In Pfizers trial, all four participants who experienced Bells palsy received the vaccine. These are 3 of the 4 volunteers who developed Bells palsy after being vaccinated with the Pfizer (SIC) covid experimental vaccine. As the United Kingdom began administering people with the Pfizer-BioNTech vaccine, four people who got Pfizer's coronavirus vaccine in the firm's trial developed Bell's palsy, a form of temporary facial paralysis, according to US regulators' report on the shot. Ill go on to say there were deaths in the trial and none of them were linked to the vaccine. To be clear, four participants in the Pfizer vaccine trial and four participants in Modernas trial did experience Bells palsy.",
        "\"We strongly believe the vaccine distribution process could begin as soon as the week of December 14,\" Pence said, according to audio of the call obtained by CBS News. We strongly believe the vaccine distribution process could begin as soon as the week of Dec. 14, Pence said on the call, according to CBS News. Siphiwe Sibeko, Associated Press Vice President Mike Pence recently told governors across the country that some doses of the COVID-19 vaccine could be distributed in two weeks, CBS News reports.",
        "Overall, positive baseline anti-spike antibodies were associated with lower rates of PCR-positivity (with or without symptoms) (adjusted rate ratio 0.24 [95%CI 0.08-0.76, p=0.015]). Conclusions Prior SARS-CoV-2 infection that generated antibody responses offered protection from reinfection for most people in the six months following infection."
    ],
    "augmented_text": [
        "In Pfizer's clinical trial, all four individuals who encountered Bell's palsy received the vaccine. These include three out of the four volunteers who developed Bell's palsy after receiving the experimental COVID vaccine from Pfizer. As the United Kingdom initiated the administration of the Pfizer-BioNTech vaccine, four participants in Pfizer's vaccine trial reported cases of Bell's palsy, a temporary facial paralysis, as documented in the regulatory report from US health authorities. It's important to note that there were deaths recorded during the trial; however, none of these fatalities were linked to the vaccine. It's noteworthy that four participants in both Pfizer's and Moderna's vaccine trials experienced incidents of Bell's palsy.",
        "\"We have a strong belief that the commencement of the vaccine distribution process could initiate as early as the week of December 14,\" Pence conveyed during a call, as revealed by audio obtained by CBS News. This statement echoes Vice President Mike Pence's recent communication to governors nationwide, where he suggested that certain doses of the COVID-19 vaccine might be distributed within the next two weeks, according to a report by CBS News, citing Siphiwe Sibeko of the Associated Press.",
        "In summary, individuals with positive baseline anti-spike antibodies exhibited lower rates of PCR-positivity, whether symptomatic or asymptomatic (adjusted rate ratio 0.24 [95% CI 0.08-0.76, p=0.015]). The findings lead to the conclusion that a previous SARS-CoV-2 infection, resulting in the generation of antibody responses, conferred protection against reinfection for the majority of individuals within the six months following the initial infection."
    ]
}

ecqa_data_augmentation_demonstrations = {
    "original_text": [
        "Where do you find wild cats?",
        "The robber wanted to conceal his pistol, where did he put it?",
        "What are students trying to do?"
    ],
    "augmented_text": [
        "Where can one encounter wild felines?",
        "The thief aimed to hide his firearm; where did he choose to conceal it?",
        "What objectives are students attempting to achieve?"
    ]
}


def get_augmentation_prompt_by_demonstrations(ds, first_field, second_field):
    if ds == "covid_fact":
        if first_field is not None:
            temp = first_field
            dictionary = covid_fact_claim_data_augmentation_demonstrations
        else:
            temp = second_field
            dictionary = covid_fact_evidence_data_augmentation_demonstrations
    else:
        dictionary = ecqa_data_augmentation_demonstrations
        temp = first_field

    prompt = "Each 2 items in the following list contains the original text and augmented. " \
             "Your task is to generate a new augmented text based on the given input.\n"

    for i in range(len(dictionary["original_text"])):
        prompt += f"Original text: {dictionary['original_text'][i]}\nAugmented text: {dictionary['augmented_text'][i]}\n\n"

    prompt += f"Original text: {temp}\nAugmented text:"

    return prompt


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
