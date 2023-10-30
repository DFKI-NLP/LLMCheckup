# -*- coding: utf-8 -*-

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

import pandas as pd
path = "/content/drive/MyDrive/GPT-4_rationales_OLID_val_132.csv"

df = pd.read_csv(path)

prompts = list(df["prompt"])[:40]
completions = list(df["completion"])[:40]

instances = []

for i in range(len(prompts)):
    instances.append(prompts[i].split("\r\nBased on the tweet, the predicted label is non-offensive. Without revealing the predicted label in your response, explain why: ")[0].split("Tweet:")[1][2:-1])

labels = [0 for i in range(len(prompts))]

new_rationale = [
    "The tweet contains strong language that can be considered offensive to some individuals due to its explicit and coarse nature. The use of the profanity is meant to emphasize a point, but it also has the potential to offend or upset people who find such language inappropriate or disrespectful. While the tweet might be intended as humor or a commentary on the challenges of adulthood, the inclusion of offensive language could make it offensive to a portion of the audience.",
    "The tweet contains derogatory language ('weirdos') and a negative judgment ('think about sex in a very weird way') directed towards a specific group of people. It's making a negative generalization about individuals who might have certain thoughts or behaviors related to sex. This could potentially be offensive to those who identify with or are associated with the group being criticized. The language used in the tweet contributes to a negative and judgmental tone, which aligns with characteristics often seen in offensive content.",
    "The tweet contains derogatory language ('beta liberals'), which is aimed at demeaning a particular group of people based on their political beliefs. Additionally, the use of 'lmao' (laughing my ass off) can be seen as mocking or dismissive, and the phrase 'you can cry all day to yourselves' is belittling and condescending in tone. The overall language and tone of the tweet are likely to be offensive to the targeted group, as it's meant to ridicule and insult their beliefs while implying that they are overly sensitive or emotional.",
    "The tweet contains derogatory language and a negative tone towards a specific individual. It refers to a political figure in a disrespectful manner, using a distorted version of their name. Additionally, it accuses this individual of using diversion tactics, which can be seen as a form of criticism. The use of terms like 'shit' and the overall accusatory tone contribute to the potentially offensive nature of the tweet. The tweet's intent seems to be to express strong dissatisfaction with the mentioned individual's actions and implies a level of deception or manipulation.",
    "The tweet contains offensive language that refers to someone's posterior in a derogatory manner. It also includes a potentially offensive term 'ass,' which can be considered disrespectful or vulgar. Additionally, the use of this language may be seen as impolite or inappropriate when addressing someone by referring to their age. The tweet combines these elements in a way that is likely to be offensive to others.",
    "The tweet appears to contain negative and critical sentiments towards Democrats, accusing them of covering up issues. It also dismisses the notion that people believe they have solutions, followed by a sarcastic tone with statements about gun control, abortion control, free medical care, and education. The tone and language used in the tweet could be seen as offensive to some due to its confrontational and dismissive nature. It appears to criticize political ideologies and policies while also using sarcasm to undermine certain social issues.",
    "The tweet contains negative and confrontational language directed towards a group of people ('@USER @USER @USER @USER @USER @USER @USER @USER'). The use of phrases like 'blindly ignorant,' 'shameful people,' and 'deep thinking you lack' is not only critical but also potentially derogatory. The tweet also implies that these individuals are not capable of making informed decisions, which can be seen as offensive and disrespectful. The overall tone of the tweet seems to be attacking and demeaning, which is likely to be perceived as offensive by many readers.",
    "The tweet includes a derogatory term ('BALLS') and discusses a controversial political movement ('ANTI-ANTIFA'). The language used seems to be inflammatory and disrespectful, as it employs a vulgar term and appears to make a negative statement about a specific group of people. The tweet could potentially incite or provoke negative reactions from individuals who are associated with or supportive of the movement mentioned. Overall, the tweet appears to be intended to offend or express strong disapproval.",
    "The tweet contains derogatory language and a negative characterization of a political group ('Liberals'). It uses offensive language ('rats ass'), and makes a sweeping generalization about their motives and behaviors. Additionally, it attributes negative qualities to this group by suggesting they only care about power and criticize everything that doesn't align with their views. The language and tone used in the tweet are likely to be perceived as offensive and disrespectful towards the mentioned political group.",
    "The tweet contains derogatory language and a negative assessment of the individual mentioned (Trumpster), using a pejorative term. It also makes a critical statement about the individual's actions, implying incompetence or inadequacy. The tone of the tweet appears to be mocking and disparaging. As a result, it comes across as offensive due to the derogatory language and the negative sentiment expressed towards the individual.",
    "The tweet uses language that implies taking advantage of someone who has received money, potentially in a deceptive or manipulative manner. The phrase 'Alaye turn if your maga don pay' suggests that the person is advising someone to take action once they've successfully extracted money from their victim ('maga'). This implies a certain level of dishonesty or exploitation, which could be seen as offensive or negative behavior. The tweet seems to promote actions that might be considered morally questionable, which could lead to offense for some readers.",
    "The tweet contains negative judgments and potentially offensive language towards an individual. It suggests that the person mentioned should not be allowed to own a gun, implying they might be dangerous or mentally unstable. Additionally, the phrase 'Such a lost soul' carries a negative connotation and could be interpreted as derogatory. The statement that 'All he Needs is Jesus Christ' also seems to make a judgment about the person's beliefs or character in a potentially offensive way. The use of the cross emoji at the end could be seen as disrespectful or insensitive, depending on the context. Overall, the tweet includes derogatory remarks and judgments about an individual, which is likely why it is predicted to be offensive.",
    "The tweet contains strong language and negative emotions directed towards the individuals mentioned in it. The phrase 'laughing into her face and almost pissing their pants' suggests that the people being talked about were mocking or ridiculing someone, which can be seen as offensive behavior. Additionally, the use of 'wtf' (an acronym for a strong expletive) and the phrase 'mind ur own business' reflect frustration and anger, contributing to the overall negative tone of the tweet. These elements collectively create an offensive context within the tweet.",
    "The tweet contains a negative characterization of a political group ('negative Democrat') and uses a tone that implies a dismissive attitude ('refuses to disappear'). This kind of language can be seen as derogatory and disrespectful towards the mentioned political group. Additionally, the term 'disappear' suggests a desire for the individual to be removed or silenced, which can be interpreted as offensive or hostile. The tweet uses partisan language and a critical tone, potentially contributing to a divisive and confrontational discourse.",
    "The tweet contains offensive language directed towards a specific subject, in this case, 'battlefield URL.' The use of explicit language in the form of a curse word and the tone of the message indicate a negative sentiment and potential frustration or anger towards the mentioned subject. This type of language and tone can be perceived as offensive or aggressive, particularly when directed at someone or something in a derogatory manner.",
    "The tweet contains a desire to win in order to obtain a good PC and engage in a specific activity. However, it also includes a casual reference to 'shit posting' and uses a slang term ('vrc') that might not be universally understood. The phrase 'shit post' can be considered offensive as it involves using derogatory language. Additionally, the mention of 'full body' in this context might allude to explicit or inappropriate content. While the offensive nature of the tweet might not be extreme, it does include elements that could be perceived as offensive or inappropriate by some individuals.",
    "The tweet appears to reference a political figure and a specific political slogan in a concise manner. The tone of the tweet seems to be expressing a viewpoint related to this figure and their associated slogan. Political discussions, especially those involving polarizing figures, can often elicit strong emotions and lead to offensive or heated exchanges. In this case, the tweet seems to be making a simple statement, but the potential for offense likely arises from the polarizing nature of the political figure and the slogan mentioned, which can easily spark contentious debates and offensive responses from those who hold differing opinions.",
    "The tweet contains a derogatory term ('fool') directed at an individual ('@USER'). The term used could be considered offensive as it's meant to belittle or insult the person. It's important to use respectful and considerate language when addressing others, and the tweet seems to deviate from that norm by using a term that is demeaning.",
    "The tweet includes a negative judgment directed at a specific individual, Bono. The choice of the word 'sucks' is derogatory and expresses strong disapproval or criticism towards the person in question. This kind of language can be considered offensive as it's intended to demean or belittle the individual, potentially provoking a negative emotional response.",
    "The tweet contains potentially offensive language, as it includes a profanity. The word used in the tweet can be considered vulgar and offensive to some people. While the tweet seems to be discussing a meta-joke about the impact of the original tweet on calculations, the choice of words adds a tone that could be offensive or inappropriate in certain contexts.",
    "The tweet contains potentially offensive language as it connects the topic of gun control with the term 'terrorists.' The use of the term 'terrorists' can be sensitive and offensive because it associates a specific group of people with violence and harm. This kind of language can be considered offensive as it oversimplifies a complex issue and potentially makes negative assumptions about a particular group. It's important to approach discussions around serious topics like gun control with respectful and balanced language to foster constructive conversations.",
    "The tweet contains offensive language and potentially offensive content. It includes a racial slur and references to violence ('gatlin gun' and 'n****s that sit on the corner like boxers between rounds'). The racial slur is derogatory and inappropriate, and the mention of violence could be seen as promoting harmful behavior. The overall tone and content of the tweet are likely to be offensive and inappropriate for many readers."
    "The tweet contains offensive language and a confrontational tone. It uses derogatory terms ('petty ass larries') to refer to a specific group of people, which can be considered disrespectful and offensive. Additionally, it includes an expletive ('trash talked') and an aggressive phrase ('stay out of our mention eh?!'), both of which contribute to the offensive nature of the tweet. The language used in the tweet is likely to provoke or escalate conflict rather than promote constructive communication.",
    "The tweet contains a statement that could be interpreted as a derogatory comment about an individual's mental state. While it doesn't use explicit offensive language, it does make a negative and potentially hurtful assessment of someone's mental well-being. Such remarks can be considered offensive as they might be seen as disrespectful or demeaning towards the person being discussed.",
    "The tweet contains a derogatory term ('hypocritical hack') directed at a specific individual ('@USER'). The term 'hypocritical hack' is a negative judgment that combines accusations of being insincere or contradictory with a demeaning term. The overall tone of the tweet appears confrontational and disrespectful, as it accuses the person of being dishonest or lacking integrity. Additionally, the inclusion of a URL might suggest that the tweet is meant to amplify negative sentiment or criticism towards the mentioned individual, potentially leading to offensive or harmful content.",
    "The tweet contains elements that could be considered offensive. It includes the accusation of someone being a 'liar,' which is a direct attack on their honesty and integrity. Additionally, the use of 'IK' without context makes it unclear who the tweet is referring to, but it's clear that the intention is to criticize or insult this person. The language used is also quite blunt and confrontational, which contributes to the overall offensive tone of the tweet.",
    "The tweet uses derogatory language to describe someone as \"bitchy,\" which is an offensive term. It also appears to be directed at a specific user, potentially with the intent to insult or criticize them. The tone of the tweet is confrontational and accusatory, suggesting frustration or irritation. Additionally, it questions why the user is not responding, which implies dissatisfaction or impatience with the other person. Overall, the tweet contains elements of rudeness and negativity, which contribute to its potentially offensive nature.",
    "This tweet contains offensive language, specifically the use of a racial slur, which is highly inappropriate and offensive to many people. The tweet also implies a threat to \"throw the whole friend away\" if they don't provide the desired level of support or enthusiasm. This combination of offensive language and the implication of discarding a friend in a negative context makes the tweet offensive.",
    "The tweet contains a serious and highly sensitive accusation against specific individuals, including the Pope, suggesting that they should be prosecuted for allegedly covering up and protecting child rapists. This statement is not only accusatory but also involves a grave and emotionally charged subject matter. Accusing someone of such a serious crime without concrete evidence can be seen as offensive, defamatory, and harmful. It's important to approach discussions about such sensitive topics with caution and respect for due process and the presumption of innocence.",
    "The tweet appears to be making a negative and critical comment about someone, referring to their alleged involvement in corruption and dishonesty. It directly mentions another Twitter user and accuses them of engaging in these unethical behaviors. While expressing opinions or criticisms on social media is common, the tone and content of this tweet seem to be directed towards attacking or disparaging someone, which can be considered offensive or hurtful.",
    "The tweet contains the term \"soy boy,\" which is often used as a derogatory slang term to criticize or mock someone's masculinity or perceived lack thereof. While it may not contain explicit profanity or hate speech, it is considered offensive as it's intended to belittle or insult the person being referred to. It's a form of name-calling that is meant to be derogatory and demeaning, which is why it's generally categorized as offensive language.",
    "The tweet contains a request to \"kick McConnell in the butt\" directed at the user's political opposition. While it doesn't use explicit offensive language, it does employ a metaphorical expression that suggests physical harm or aggression towards a political figure. This type of language can be seen as aggressive or disrespectful in a political context and may be considered offensive by some people, particularly if it promotes or condones physical harm or aggression towards individuals with differing political views.",
    "The tweet contains potentially offensive language. Specifically, it uses a term that could be considered vulgar or inappropriate in many professional or social contexts (\"sick ass\"). This language may be offensive to some people as it is slang and can be seen as disrespectful or unprofessional. It's important to be mindful of the language we use on social media to ensure that it is respectful and appropriate for the audience and context.",
    "The tweet contains explicit language related to bodily functions, which is generally considered offensive in many social contexts. It uses profanity to describe a personal experience, which can be seen as crude and inappropriate for public discussion. The content of the tweet is likely to be offensive to many people due to its explicit nature, as it discusses a topic that is typically considered private and not suitable for public discourse.",
    "The tweet contains a confrontational tone with phrases like \"do not think so\" and \"maybe do a little research buddy!\" It addresses the user with the term \"buddy,\" which can be seen as condescending or dismissive in certain contexts. The use of \"maybe do a little research\" implies that the user hasn't done their homework or lacks knowledge, which can come across as offensive or insulting, depending on the overall tone and intention of the conversation. Overall, the language and tone in this tweet suggest that it is potentially offensive.",
    "The tweet appears to be making a speculative and potentially offensive statement. It mentions the idea of finding a picture of someone named Bret holding a woman in a certain way and suggests that this would cause a reaction from liberals, implying that they would \"go insane.\"",
    "The tweet uses strong political language and acronyms like \"tcot\" (Top Conservatives on Twitter) and \"MAGA\" (Make America Great Again), which are associated with a specific political ideology. It also includes the phrase \"Complacency kills,\" which, in the context of politics, can be interpreted as a call to action and a warning about the consequences of not participating in the electoral process. However, it does not contain explicit offensive language or directly target individuals or groups in a derogatory manner. Instead, it seems to be a passionate political message urging people to vote in an upcoming election.",
    "The tweet contains offensive language, specifically a profanity directed towards another user. This kind of language is considered offensive and disrespectful in most online and offline interactions, and it violates the guidelines and rules of many social media platforms, including Twitter. Therefore, it is likely to be classified as offensive by content moderation algorithms or human reviewers.",
    "This tweet could be considered offensive because it contains a negative statement directed at the user, implying that they are not welcome or unwanted. It can be seen as a form of online harassment or bullying, as it conveys a hostile and unwelcoming tone towards the targeted individual. Such messages can contribute to a negative online environment and potentially harm the mental well-being of the recipient.",
    "The tweet can be considered offensive because it contains explicit language (\"fix this shit\") and appears to be directed in a hostile or aggressive manner toward the mentioned users. It lacks constructive criticism or a polite tone, which often characterizes offensive or disrespectful communication on social media platforms like Twitter. The use of strong, derogatory language and a confrontational approach can be offensive to others who come across the tweet.",
    "The tweet may be considered offensive because it potentially engages in cyberbullying or harassment. It mentions two specific users, presumably in a context where the intention could be to mock or ridicule them. It's essential to be respectful and considerate when using social media, and targeting individuals in a negative or derogatory manner goes against those principles.",
    "The tweet can be considered offensive because it contains several elements that contribute to a potentially offensive tone: Excessive mentions: The use of \"@USER @USER @USER @USER @USER\" appears to be an attempt to flood or overwhelm specific individuals on Twitter. This can be seen as a form of harassment or an attempt to annoy or provoke those mentioned. Negative assumption: The tweet makes a negative assumption about the mentioned users, suggesting that they are \"incapable of understanding contradictions.\" This assumption can be interpreted as an insult or a demeaning statement. Lack of constructive content: The tweet does not provide any constructive feedback or engage in meaningful discussion. Instead, it seems to be aimed at belittling or criticizing the mentioned users. Overall, the tweet's tone and content contribute to a potentially offensive message, which is why it could be labeled as offensive without revealing the specific label.",
    "The tweet contains offensive language, which is evident from the use of profanity (\"shit\"). This type of language is generally considered offensive and inappropriate for many social media platforms, including Twitter, as it can contribute to a negative and hostile online environment. Therefore, based on the presence of offensive language, it is likely that the predicted label for this tweet is indeed \"offensive.\"",
]

temp = [
    "@USER Yeahh tbh most that do are weirdos and think about sex in a very weird way.",
    "@USER @USER Here come the beta liberals lmao *convo muted you can cry all day to yourselves.",
    "@USER @USER When tRUmp posts shit like this you know that he is distracting from something else that is going on. Distractification",
    "@USER @USER My old ass was here URL",
    "@USER Democrats haven't done anything but cover up ..so all you think you have a solution .ya don't.....you want gun control.? How about abortion control. Free medical. Free ed. Quit tripping",
    "@USER @USER @USER @USER @USER @USER @USER @USER How blindly ignorant you are. Shameful people with the deep thinking you lack can vote",
    "ANTI-ANTIFA IS BALLS",
    "@USER @USER Liberals don't give a rats ass but having power!! Period!! They cry over everything that doesn't go their way",
    "@USER Well if Trumpster said it then it must be true he also have himself a 10 on handling Cuba hurricane sad piece of something he is...",
    "Alaye turn if your maga don pay",
    "@USER @USER He is one who should Not own a gun-Such a lost soul—All he Needs is Jesus Christ :latin_cross:",
    "@USER bro they were laughing into her face and almost pissing their pants and i was standing beside it like wtf mind ur own business",
    "@USER Another negative Democrat who refuses to disappear!",
    "@USER K nvm fuck battlefield URL",
    "@USER same i just want to win so i can finally have a good pc and shit post in full body in vrc",
    "@USER @USER Trump is maga",
    "@USER He really bisca fool huh",
    "@USER Bono sucks",
    "@USER A whole meta joke here would be my original tweet fuckin up the calculations.",
    "@USER Gotta keep that gun control law to get those terrorists",
    "@USER He also said he had a gatlin gun and niggas that sit on the corner like boxers between rounds",
    "@USER Zarries didn’t start this conversation! It was a larry shipper that trash talked our ship!! So how about y’all petty ass larries stay out of our mention eh?!!",
    "@USER @USER He is losing his mind",
    "@USER you are a hypocritical hack. URL",
    "@USER IK never did. He is liar.",
    "@USER He is too bitchy... And you?! I ask a question and you dont respond Why would this be 1 way street?",
    "@USER If you ain’t hyping me up like the nigga in the back ima have to throw the whole friend away",
    "@USER @USER I think the pope and some others should be prosecuted for covering up and protecting child rapists!",
    "@USER He is most impressive at corruption and lies.",
    "@USER @USER Actually he is a soy boy.",
    "@USER Thank you for your fortitude and perseverance. Please give McConnell a kick in the butt from some of us conservatives.",
    "@USER My coworker has a sick ass one for work",
    "@USER @USER I SHITTED/SHAT YTD AND LOST 0.6KG CAN U BELIEVE IT",
    "@USER Do not think so maybe do a little research buddy!",
    "@USER @USER @USER No just imagining if we found a picture of Bret holding a woman like this. Liberals will go insane",
    "tcot MAGA Complacency kills. Vote in November URL",
    "@USER Fuck you it’s 3",
    "@USER you are not welcome",
    "@USER @USER fix this shit!",
    "@USER @USER when robyn posts a selfie the world stops for a moment to appreciate it",
    "@USER @USER @USER @USER @USER I know you are not being facetious. You are incapable of understanding contradictions.",
    "@USER Awh shit.",

]

completions += new_rationale
labels += [1 for i in range(len(new_rationale))]

print(len(completions))
print(len(labels))

max_len = 0
for sent in completions:

    # 将文本分词，并添加 `[CLS]` 和 `[SEP]` 符号
    input_ids = tokenizer.encode(sent, add_special_tokens=True)
    max_len = max(max_len, len(input_ids))

print('Max sentence length: ', max_len)

import torch

input_ids = []
attention_masks = []

for sent in completions:
    encoded_dict = tokenizer.encode_plus(
                        sent,                      # 输入文本
                        add_special_tokens = True, # 添加 '[CLS]' 和 '[SEP]'
                        max_length = 256,           # 填充 & 截断长度
                        pad_to_max_length = True,
                        return_attention_mask = True,   # 返回 attn. masks.
                        return_tensors = 'pt',     # 返回 pytorch tensors 格式的数据
                   )

    # 将编码后的文本加入到列表
    input_ids.append(encoded_dict['input_ids'])

    # 将文本的 attention mask 也加入到 attention_masks 列表
    attention_masks.append(encoded_dict['attention_mask'])

# 将列表转换为 tensor
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(labels)

# 输出第 1 行文本的原始和编码后的信息
print('Original: ', completions[0])
print('Token IDs:', input_ids[0])

from torch.utils.data import TensorDataset, random_split

# 将输入数据合并为 TensorDataset 对象
dataset = TensorDataset(input_ids, attention_masks, labels)

# 计算训练集和验证集大小
train_size = int(0.95 * len(dataset))
val_size = len(dataset) - train_size

# 按照数据大小随机拆分训练集和测试集
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

print('{:>5,} training samples'.format(train_size))
print('{:>5,} validation samples'.format(val_size))

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

# 在 fine-tune 的训练中，BERT 作者建议小批量大小设为 16 或 32
batch_size = 8

# 为训练和验证集创建 Dataloader，对训练样本随机洗牌
train_dataloader = DataLoader(
            train_dataset,  # 训练样本
            sampler = RandomSampler(train_dataset), # 随机小批量
            batch_size = batch_size # 以小批量进行训练
        )

# 验证集不需要随机化，这里顺序读取就好
validation_dataloader = DataLoader(
            val_dataset, # 验证样本
            sampler = SequentialSampler(val_dataset), # 顺序选取小批量
            batch_size = batch_size
        )

from transformers import BertForSequenceClassification, AdamW, BertConfig

# 加载 BertForSequenceClassification, 预训练 BERT 模型 + 顶层的线性分类层
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", # 小写的 12 层预训练模型
    num_labels = 2, # 分类数 --2 表示二分类
                    # 你可以改变这个数字，用于多分类任务
    output_attentions = False, # 模型是否返回 attentions weights.
    output_hidden_states = False, # 模型是否返回所有隐层状态.
    return_dict=False
)

model.cuda()

optimizer = AdamW(model.parameters(),
                  lr = 2e-6, # args.learning_rate - default is 5e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8
                )

from transformers import get_linear_schedule_with_warmup

# 训练 epochs。 BERT 作者建议在 2 和 4 之间，设大了容易过拟合
epochs = 1

# 总的训练样本数
total_steps = len(train_dataloader) * epochs

# 创建学习率调度器
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = 0,
                                            num_training_steps = total_steps)

import numpy as np

# 根据预测结果和标签数据来计算准确率
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

import time
import datetime

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # 四舍五入到最近的秒
    elapsed_rounded = int(round((elapsed)))

    # 格式化为 hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

import random
import numpy as np

# 以下训练代码是基于 `run_glue.py` 脚本:
# https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

device = "cuda" if torch.cuda.is_available() else "cpu"

# 设定随机种子值，以确保输出是确定的
seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# 存储训练和评估的 loss、准确率、训练时长等统计指标,
training_stats = []

# 统计整个训练时长
total_t0 = time.time()

for epoch_i in range(0, epochs):

    # ========================================
    #               Training
    # ========================================


    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    # 统计单次 epoch 的训练时间
    t0 = time.time()

    # 重置每次 epoch 的训练总 loss
    total_train_loss = 0

    # 将模型设置为训练模式。这里并不是调用训练接口的意思
    # dropout、batchnorm 层在训练和测试模式下的表现是不同的 (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
    model.train()

    # 训练集小批量迭代
    for step, batch in enumerate(train_dataloader):

        # 每经过40次迭代，就输出进度信息
        if step % 40 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        # 准备输入数据，并将其拷贝到 gpu 中
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        # 每次计算梯度前，都需要将梯度清 0，因为 pytorch 的梯度是累加的
        model.zero_grad()

        # 前向传播
        # 文档参见:
        # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
        # 该函数会根据不同的参数，会返回不同的值。 本例中, 会返回 loss 和 logits -- 模型的预测结果
        loss, logits = model(b_input_ids,
                             token_type_ids=None,
                             attention_mask=b_input_mask,
                             labels=b_labels)
        print(loss)
        # 累加 loss
        total_train_loss += loss.item()

        # 反向传播
        loss.backward()

        # 梯度裁剪，避免出现梯度爆炸情况
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # 更新参数
        optimizer.step()

        # 更新学习率
        scheduler.step()

    # 平均训练误差
    avg_train_loss = total_train_loss / len(train_dataloader)

    # 单次 epoch 的训练时长
    training_time = format_time(time.time() - t0)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(training_time))

    # ========================================
    #               Validation
    # ========================================
    # 完成一次 epoch 训练后，就对该模型的性能进行验证

    print("")
    print("Running Validation...")

    t0 = time.time()

    # 设置模型为评估模式
    model.eval()

    # Tracking variables
    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0

    # Evaluate data for one epoch
    for batch in validation_dataloader:

        # 将输入数据加载到 gpu 中
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        # 评估的时候不需要更新参数、计算梯度
        with torch.no_grad():
            (loss, logits) = model(b_input_ids,
                                   token_type_ids=None,
                                   attention_mask=b_input_mask,
                                   labels=b_labels)

        # 累加 loss
        total_eval_loss += loss.item()

        # 将预测结果和 labels 加载到 cpu 中计算
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # 计算准确率
        total_eval_accuracy += flat_accuracy(logits, label_ids)


    # 打印本次 epoch 的准确率
    avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
    print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

    # 统计本次 epoch 的 loss
    avg_val_loss = total_eval_loss / len(validation_dataloader)

    # 统计本次评估的时长
    validation_time = format_time(time.time() - t0)

    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))

    # 记录本次 epoch 的所有统计信息
    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. Accur.': avg_val_accuracy,
            'Training Time': training_time,
            'Validation Time': validation_time
        }
    )


"""
Evaluation
"""
print("Training complete!")
print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

# 0.6633333333333333
path = "/content/drive/MyDrive/rationales/rationalization_few_shot_gpt-j-6b.json"
# 0.64
# path = "/content/drive/MyDrive/rationales/rationalization_few_shot_gpt-neo-2.7b.json"
# 0.6066666666666667
# path = "/content/drive/MyDrive/rationales/rationalization_plan_and_solve_gpt-j-6b.json"

# 0.6233333333333333
# path = "/content/drive/MyDrive/rationales/rationalization_plan_and_solve_gpt-neo-2.7b.json"

# 0.65
# path = "/content/drive/MyDrive/rationales/rationalization_zero_cot_gpt-j-6b.json"

# 0.62
# path = "/content/drive/MyDrive/rationales/rationalization_zero_cot_gpt-neo-2.7b.json"

import json
fileObject = open(path, "r")
jsonContent = fileObject.read()
json_list = json.loads(jsonContent)

json_list[2]["rationalization"]

idx = []
results = []
for i in range(len(json_list)):
  if json_list[i]["rationalization"] != "error":
    results.append(json_list[i]["rationalization"])
  else:
    idx.append(i)

idx

predictions = []
for i in results:
  encoded_dict = tokenizer.encode_plus(
      i,                      # 输入文本
      add_special_tokens = True, # 添加 '[CLS]' 和 '[SEP]'
      max_length = 512,           # 填充 & 截断长度
      # pad_to_max_length = True,
      return_attention_mask = True,   # 返回 attn. masks.
      return_tensors = 'pt',     # 返回 pytorch tensors 格式的数据
  )
  input_ids = encoded_dict["input_ids"]
  attention_mask = encoded_dict['attention_mask']

  input_ids = input_ids.to(device)
  attention_mask = attention_mask.to(device)
  input_model = {
      'input_ids': input_ids.long(),
      'attention_mask': attention_mask.long(),
  }
  output_model = model(**input_model)[0]
  predictions.append(torch.argmax(output_model).item())

df = pd.read_csv("/content/drive/MyDrive/offenseval_val.csv")
labels = np.array(list(df["label"])[:300])

print(len(predictions))
print(len(labels))

print(np.sum(predictions==labels)/len(labels))
#
# !huggingface-cli login
#
# model.push_to_hub("qiaw99/olid-rationale-sim-bert-base-uncased", create_pr=1)



