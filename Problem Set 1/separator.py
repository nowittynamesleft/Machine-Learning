#Separates the file into the first 4000 emails and the next 1000 emails
#The newlines are easily seen using a text editor since the original document does not contain any newlines

text = open("spam_train.txt").read()
output = open("newspam_train.txt","w")
count = 0;


for s in text.split():
    if s == '1' or s == '0':
        count = count + 1
        if count == 4000:
            output.write("\n\n")
    output.write(" " + s);
