import pickle
import spacy
import pathlib
from spacy import displacy

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

ner = pickle.load(open('Model_pickle.pkl', 'rb'))

abc = ner(''' Shahid Afridi is 53 years old. Symptoms are highly variable and may include, persistent lump, weight loss and other unexplained changes in the body.A lump or mass in the breast that feels different from the surrounding tissue causes liver cancer,can be treated with Aspirin (650mg) tablets ith 120 mmHg , 120cm , 5ft5in, 100kg, 150 bpm, 90%, 9589408732, 583201, 93.2 °F, 22°c ''')  # input sample text

print(spacy.displacy.serve(abc, style="ent"))
