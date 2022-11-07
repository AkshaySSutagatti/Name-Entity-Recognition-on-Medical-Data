from flask import Flask, render_template, url_for, request
import pickle
import spacy
import pathlib
from spacy import displacy
import webbrowser

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# load the model from disk
# filename = 'Model_pickle.pkl'
ner = pickle.load(open('Model_pickle.pkl', 'rb'))
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    #	df= pd.read_csv("spam.csv", encoding="latin-1")
    #	df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
    #	# Features and Labels
    #	df['label'] = df['class'].map({'ham': 0, 'spam': 1})
    #	X = df['message']
    #	y = df['label']
    #
    #	# Extract Feature With CountVectorizer
    #	cv = CountVectorizer()
    #	X = cv.fit_transform(X) # Fit the Data
    #
    #    pickle.dump(cv, open('tranform.pkl', 'wb'))
    #
    #
    #	from sklearn.model_selection import train_test_split
    #	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    #	#Naive Bayes Classifier
    #	from sklearn.naive_bayes import MultinomialNB
    #
    #	clf = MultinomialNB()
    #	clf.fit(X_train,y_train)
    #	clf.score(X_test,y_test)
    #    filename = 'nlp_model.pkl'
    #    pickle.dump(clf, open(filename, 'wb'))

    # Alternative Usage of Saved Model
    # joblib.dump(clf, 'NB_spam_model.pkl')
    # NB_spam_model = open('NB_spam_model.pkl','rb')
    # clf = joblib.load(NB_spam_model)

    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = ner(str(data))
        webbrowser.open('http://localhost:5001')
        spacy.displacy.serve(vect, style="ent",port=5001)
    #return render_template('result.html')


if __name__ == '__main__':
    app.run(debug=True)
