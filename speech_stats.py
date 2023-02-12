import pandas as pd
from collections import Counter

def make_speech_statistics():
    """ Genenerate speech """
    columnNames= ['Parliament', 'speechID', 'speechtext', 'speakerID', 'speakername', 'chamber', 'riding', 'speakerparty', 'majorityparty', 'province']
    result = pd.read_csv('./data/38to42Parl_forembeddings.csv', sep='\t', header=None, names=columnNames)

    # name that matches output from embeddings: MP_party_parliamentNo
    result['name_parl'] = result.apply(lambda row: row.speakername+'_'+row.speakerparty+'_'+str(row.Parliament), axis=1)

    # add column for number of words in each speech
    result['speech_length'] = result.speechtext.str.count('\s+')

    # for each MP, get the total number of speeches made and the total speech volume in number of words
    speechStats = pd.DataFrame.from_dict(Counter(result.name_parl), orient='index', columns=['speechFrequency'])
    speechStats['totalSpeechLength'] = [0]*len(speechStats)
    for i in range(len(result)):
        speechStats.loc[result.name_parl[i],'totalSpeechLength'] += result.speech_length[i]

    speechStats = speechStats.reset_index()

    speechStats.to_csv('speechStats.csv',sep=',', index=False, line_terminator='\n', header=False)


if __name__ == '__main__':
    make_speech_statistics()