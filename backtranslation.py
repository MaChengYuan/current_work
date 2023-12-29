from transformers import MarianMTModel, MarianTokenizer
from backtranslation import *
import os
from tqdm import tqdm

first_model_name = 'Helsinki-NLP/opus-mt-en-ROMANCE'

# Get the tokenizer
first_model_tkn = MarianTokenizer.from_pretrained(first_model_name)

# Load the pretrained model based on the name
first_model = MarianMTModel.from_pretrained(first_model_name)


second_model_name = 'Helsinki-NLP/opus-mt-ROMANCE-en'

# Get the tokenizer
second_model_tkn = MarianTokenizer.from_pretrained(second_model_name)

# Load the pretrained model based on the name
second_model = MarianMTModel.from_pretrained(second_model_name)
def format_batch_texts(language_code, batch_texts):
  
    formated_bach = [">>{}<< {}".format(language_code, text) for text in batch_texts]

    return formated_bach
    
def perform_translation(batch_texts, model, tokenizer, language="fr"):
    # Prepare the text data into appropriate format for the model
    formated_batch_texts = format_batch_texts(language, batch_texts)
    
    # Generate translation using model
    translated = model.generate(**tokenizer(formated_batch_texts, return_tensors="pt", padding=True))

    # Convert the generated tokens indices back into text
    translated_texts = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
    
    return translated_texts
def perform_back_translation(batch_texts,sample_num):
    original_language="en"
    temporary_languages = ['fr','es','it','pt','ro','ca','gl']
    aug_sentences = []
    index = 0
    while(len(aug_sentences) != sample_num):
    # Translate from Original to Temporary Language
        tmp_translated_batch = perform_translation(batch_texts, first_model, first_model_tkn, temporary_languages[index])
        # Translate Back to English
        back_translated_batch = perform_translation(tmp_translated_batch, second_model, second_model_tkn, original_language)
        aug_sentences.append(back_translated_batch)
        tmp_translated_batch = back_translated_batch
        index += 1
        if(index == len(temporary_languages)):
            index = 0
    # Return The Final Result
    return aug_sentences

def gen_backtranslation(df , output_file, num_aug=1):

    sentence = df['sentence']
    label = df['label'].astype(str)
    
    writer = open(output_file, 'w')
    y = label
    x = sentence
    writer.write('label' + "\t" + 'sentence' + '\n')
    for i in tqdm(range(len(label))):
        
        label = y[i]
        sentence = x[i]
        aug_sentences = perform_back_translation([sentence], sample_num = num_aug)
        for aug_sentence in aug_sentences:
            aug_sentence = aug_sentence[0]
            writer.write(label + "\t" + aug_sentence + '\n')

    writer.close()
    print("generated augmented sentences with eda for to " + output_file + " with num_aug=" + str(num_aug))    

