from flair.nn import Classifier
from flair.data import Sentence

# load the model
tagger = Classifier.load('sentiment')

# make a sentence
sentence = Sentence('This movie is not at all bad.')

# predict NER tags
tagger.predict(sentence)

# print sentence with predicted tags
print(sentence)