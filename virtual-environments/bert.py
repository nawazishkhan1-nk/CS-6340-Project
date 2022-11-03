from transformers import AutoTokenizer, AutoModel

# used to tokenize the
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
# load the pre-trained model
model = AutoModel.from_pretrained("bert-base-uncased")

# example input sentence
text = "I'm excited to use bert for my project"

# encode and pass through the BERT
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
print("output=",output)
