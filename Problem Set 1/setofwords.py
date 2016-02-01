import numpy as np
training = open("newspam_train.txt").read()
wordset = {}

for s in training.split():
    if s in wordset:
        wordset[s] = wordset[s] + 1;
    else:
        wordset[s] = 1;

common = []

for word in wordset:
    if wordset[word] >= 30:
        common.append(word)

validation = open("spam_validation.txt").read()


#Transforming into feature vectors:
#Need to have a feature vector for each email
#[index of email]{word : 0/1}

#need to split input into seperate emails to evaluate emails
#emails are sets of text

#emailFeatures = np.zeros(1000, len(common)) #so the entry for 2,4
        #is 1 if the third email has the fifth word in the common word list
        #is 0 if the third email does not have the fifth word in the common word list

emails = [[] for i in range(0,2000)]
currentEmail = [] #email currently being constructed (it's a list of words)
count = 0
print(validation.split())
for word in validation.split():
    if word == '1' or word == '0':
        emails[count] = currentEmail
        currentEmail = []
        count = count + 1
        print(count)
    else:
        currentEmail.append(word)

#so now we have a list of emails in "emails", as a list of lists of words
#
