import glob
import pandas as pd
import matplotlib.pyplot as plt
import re
import string
import unicodedata
import nltk

from sklearn.feature_extraction import text as xtext 
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from nltk.tokenize import ToktokTokenizer
from nltk.stem.snowball import SnowballStemmer
from functools import reduce

def remove_unecessary_speakers(parlNo):
    """ Remove speeches without an associated name and those made by procedural speakers.
    
        Args: parliament number {38, 39, 40, 41, 42}
        Returns: None; saves output to csv 
    """
    # Load data
    files = glob.glob(f'{str(parlNo)}Parliament/**/**/*.csv')
    dfs = [pd.read_csv(fp) for fp in files]
    df = pd.concat(dfs, ignore_index=True)

    # Remove interjections or any time that speakers are not identified
    fillerProcedure = df.speakeroldname.isnull()
    df = df.loc[~fillerProcedure]

    # remove speeches when a main topic is not identified
    fillerProcedure = df.maintopic.isnull()
    df = df[~fillerProcedure]

    # remove procedural speakers not otherwise filtered
    parliamentarians = ['The Speaker', 'The Deputy Speaker', 'The Chair', 'The Assistant Deputy Speaker', 'The Deputy Chair', 'The Assistant Deputy Chair', 'The Acting Speaker']
    for person in parliamentarians:
        df = df[~df['speakeroldname'].str.contains(person)]
        
    # reset indices so that they start from 0 and go up by 1
    df = df.reset_index(drop=True)
        
    df.to_csv(f'{str(parlNo)}Parl.csv')

def strip_accents(text):
    text = unicodedata.normalize('NFD', text)
    text = text.encode('ascii', 'ignore')
    text = text.decode("utf-8")
    return str(text)

def clean_text(text):
    tk = ToktokTokenizer()
    stopwords = ['member','members','government','governments','opposition','opposite','leader',
        'hon','exminister','prime','minister','ministers','parliament','house',
        'ask','asked','asks','question','questioned','questions','bills','bill',
        'party','parties','mp','mps','sir','madam','mr','gentleman','gentlemen','lady','ladies',
        'speaker','chair','motion','motions','vote','votes','order','yes','deputy','secretary',
        'canada','canadian','canadians',
        'pursuant','supply','supplementary','please','friend','s',
        'clause','amendment','i','ii','iii','section','sections', 'colleague', 'colleagues'] + list(xtext.ENGLISH_STOP_WORDS)

    # For replacement of contractions.
    contractions = {"you'd": 'you would', "he'd": 'he would', "she's": 'she is', "where'd": 'where did', "might've": 'might have', "he'll": 'he will', "they'll": 'they will',  "mightn't": 'might not', "you'd've": 'you would have', "shan't": 'shall not', "it'll": 'it will', "mayn't": 'may not', "couldn't": 'could not', "they'd": 'they would', "so've": 'so have', "needn't've": 'need not have', "they'll've": 'they will have', "it's": 'it is', "haven't": 'have not', "didn't": 'did not', "y'all'd": 'you all would', "needn't": 'need not', "who'll": 'who will', "wouldn't've": 'would not have', "when's": 'when is', "will've": 'will have', "it'd've": 'it would have', "what'll": 'what will', "that'd've": 'that would have', "y'all're": 'you all are', "let's": 'let us', "where've": 'where have', "o'clock": 'oclock', "when've": 'when have', "what're": 'what are', "should've": 'should have', "you've": 'you have', "they're": 'they are', "aren't": 'are not', "they've": 'they have', "it'd": 'it would', "i'll've": 'i will have', "they'd've": 'they would have', "you'll've": 'you will have', "wouldn't": 'would not', "we'd": 'we would', "hadn't've": 'had not have', "weren't": 'were not', "i'd": 'i would', "must've": 'must have', "what's": 'what is', "mustn't've": 'must not have', "what'll've": 'what will have', "ain't": 'aint', "doesn't": 'does not', "we'll": 'we will', "i'd've": 'i would have', "we've": 'we have', "oughtn't": 'ought not', "you're": 'you are', "who'll've": 'who will have', "shouldn't": 'should not', "can't've": 'cannot have', "i've": 'i have', "couldn't've": 'could not have', "why've": 'why have', "what've": 'what have', "can't": 'cannot', "don't": 'do not', "that'd": 'that would', "who's": 'who is', "would've": 'would have', "there'd": 'there would', "shouldn't've": 'should not have', "y'all": 'you all', "mustn't": 'must not', "she'll": 'she will', "hadn't": 'had not', "won't've": 'will not have', "why's": 'why is', "'cause": 'because', "wasn't": 'was not', "shan't've": 'shall not have', "ma'am": 'madam', "hasn't": 'has not', "to've": 'to have', "how'll": 'how will', "oughtn't've": 'ought not have', "he'll've": 'he will have', "we'd've": 'we would have', "won't": 'will not', "could've": 'could have', "isn't": 'is not', "she'll've": 'she will have', "we'll've": 'we will have', "you'll": 'you will', "who've": 'who have', "there's": 'there is', "y'all've": 'you all have', "we're": 'we are', "i'll": 'i will', "i'm": 'i am', "how's": 'how is', "she'd've": 'she would have', "sha'n't": 'shall not', "there'd've": 'there would have', "he's": 'he is', "it'll've": 'it will have', "that's": 'that is', "y'all'd've": 'you all would have', "he'd've": 'he would have', "how'd": 'how did', "where's": 'where is', "so's": 'so as', "she'd": 'she would', "mightn't've": 'might not have'}

    # replace contractions
    text = reduce(lambda a, kv: a.replace(*kv), contractions.items(), text.lower())
    
    # replace tab, newline, and carriage return characters with spaces
    text = text.replace('\t',' ').replace('\n',' ').replace('\r',' ')
    
    # replace punctuation with spaces
    text = text.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))
    
    # remove accent characters
    text = strip_accents(text)
    
    # tokenize for stopword removal
    tokens = tk.tokenize(text)
    
    # remove stopwords
    tokens = [w for w in tokens if w not in stopwords and len(w)>2 and w!=' ' and not w.isdigit()]
    
    # stemming using Snowball
    tokens = [SnowballStemmer('english').stem(w) for w in tokens]
    
    # return string of tokens in order 
    return ' '.join(tokens)

def main():
    # load data from the Parliaments of interest (2004 to 2019)
    # note that the data already excludes speeches from the Speaker, etc., 
    # those without speakers, and those without a main topic
    root = './data/parliament_speeches'
    files = ['/42Parl.csv', '/41Parl.csv', '/40Parl.csv', '/39Parl.csv', '/38Parl.csv']
    dfs = [pd.read_csv(root+fp, low_memory=False) for fp in files]

    # remove speeches shorter than or equal to 30 words
    byParl = [df[df['speechtext'].str.count('\s+')>30] for df in dfs]

    # reset indices 
    # df = df.reset_index(drop=True)

    # save the lengths of each of the Parliaments' for use in the final dataframe construction
    lengths = [len(df) for df in byParl]

    # join the dataframes for ease of cleaning
    df = pd.concat(byParl, ignore_index=True)

    data_clean = pd.DataFrame(df.speechtext.apply(lambda x: clean_text(x)))

    # reformat for partyembed
    # 0: Congress
    # 1: Speech ID
    # 2: Raw text
    # 3: Speaker ID
    # 4: Speaker Name
    # 5: Chamber (House/Senate)
    # 6: State
    # 7: Party
    # 8: Majority Party (0/1)
    # 9: Presidential Party (0/1)

    result = pd.DataFrame()
    result['Parliament'] = [42]*lengths[0] + [41]*lengths[1] + [40]*lengths[2] + [39]*lengths[3] + [38]*lengths[4] 
    result['speechID'] = df.basepk
    result['speechtext'] = data_clean
    result['speakerID'] = df.pid
    result['speakername'] = df.speakername
    result['chamber'] = ['HoC']*len(df)
    result['riding'] = df.speakerriding
    result['speakerparty'] = df.speakerparty
    result['majorityparty'] = ['Liberal']*lengths[0] + ['Conservative']*lengths[1] + ['Conservative']*lengths[2] + ['Conservative']*lengths[3] + ['Liberal']*lengths[4] 
    result['province'] = [0]*len(df)

    result.to_csv('38to42Parl.csv',sep='\t', index=False, line_terminator='\n', header=False)


if __name__ == '__main__':
    main()


