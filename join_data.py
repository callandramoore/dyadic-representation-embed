import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pkg_resources
import glob
from gensim.models.doc2vec import Doc2Vec
from sklearn.decomposition import PCA

def generateElectionDataforRidings(rawElection, ParliamentNo):
    """ Generate df of party share, province, and MP name for each riding for a single federal election. """
    # deal with different labelling conventions across years
    label_map = {38 : ['District', 'Candidate', 'Number of Votes Percent', 'Majority Percent', 'Province'], 
          39 : ['Electoral District/Circonscription', 'Candidate/Candidat', 'Percentage of Votes Obtained /Pourcentage des votes obtenus',  'Majority Percentage/Pourcentage de majorité', 'Province'],
          40 : ['Electoral District Name/Nom de circonscription','Candidate/Candidat', 'Percentage of Votes Obtained /Pourcentage des votes obtenus', 'Majority Percentage/Pourcentage de majorité', 'Province'], 
          41 : ['Electoral District Name/Nom de circonscription','Candidate/Candidat', 'Percentage of Votes Obtained /Pourcentage des votes obtenus', 'Majority Percentage/Pourcentage de majorité', 'Province'], 
          42 : ['Electoral District Name/Nom de circonscription','Candidate/Candidat', 'Percentage of Votes Obtained /Pourcentage des votes obtenus', 'Majority Percentage/Pourcentage de majorité', 'Province']}
    ridingLabel, candidateLabel, voteObtainedLabel, majorityLabel, provinceLabel = label_map[ParliamentNo]

    # create an empty df to store election results for each riding by party 
    resultsByParty = pd.DataFrame()
    resultsByParty['riding'] = list(set(rawElection.loc[:,ridingLabel]))
    parties = ['NDP', 'Green', 'Bloc Quebecois', 'Liberal', 'Conservative', 'Other']

    # fill the df with zeros
    for p in parties:
        resultsByParty[p+'_share'] = [0]*len(resultsByParty)

    resultsByParty = resultsByParty.set_index('riding')

    # go through each candidate and put in their share of the vote in their riding and party position
    for i in range(len(rawElection)):
        if 'Liberal' in rawElection[candidateLabel][i]:
            resultsByParty.loc[rawElection[ridingLabel][i],'Liberal_share'] += rawElection[voteObtainedLabel][i]
        elif 'Conservative' in rawElection[candidateLabel][i]:
            resultsByParty.loc[rawElection[ridingLabel][i],'Conservative_share'] += rawElection[voteObtainedLabel][i]
        elif ('NDP' in rawElection[candidateLabel][i]) or ('N.D.P.' in rawElection[candidateLabel][i]):
            resultsByParty.loc[rawElection[ridingLabel][i],'NDP_share'] += rawElection[voteObtainedLabel][i]
        elif 'Bloc' in rawElection[candidateLabel][i]:
            resultsByParty.loc[rawElection[ridingLabel][i],'Bloc Quebecois_share'] += rawElection[voteObtainedLabel][i]
        elif 'Green' in rawElection[candidateLabel][i]:
            resultsByParty.loc[rawElection[ridingLabel][i],'Green_share'] += rawElection[voteObtainedLabel][i]
        else:
            resultsByParty.loc[rawElection[ridingLabel][i],'Other_share'] += rawElection[voteObtainedLabel][i]

    resultsByParty.loc[:,'province'] = [0]*len(resultsByParty)
    for i, riding in enumerate(rawElection[ridingLabel]):
        resultsByParty.loc[riding,'province'] = rawElection[provinceLabel][i]

    # extract the elected MP's name and add to the df
    # note that only the elected MP's entry has a non-null value in the 'Majority' column
    # also generate a competitvenss score which is the difference between the first and second place candidates
    resultsByParty.loc[:,'MP'] = [0]*len(resultsByParty)
    resultsByParty.loc[:,'competitiveness'] = [0.0]*len(resultsByParty)

    for i, riding in enumerate(rawElection[ridingLabel]):
        if not pd.isna(rawElection[majorityLabel][i]):
            tempname = rawElection[candidateLabel][i]
            resultsByParty.loc[riding,'MP'] = tempname
            resultsByParty.loc[riding,'competitiveness'] = rawElection[majorityLabel][i]


    # include label for Parliament Number
    resultsByParty.loc[:,'ParliamentNo'] = [ParliamentNo]*len(resultsByParty)
    resultsByParty = resultsByParty.reset_index()
    
    return resultsByParty

def main():
    # load the doc2vec model
    # MODEL_PATH = pkg_resources.resource_filename('partyembed', 'models/')
    model = Doc2Vec.load('data/38to42Parl')

    # extract vectors from the embedding model and put it in a df
    vectorlist = [model.dv[i] for i in range(0,len(model.dv.index_to_key))]
    vectordf=pd.DataFrame(vectorlist)

    # use Principle Component Analysis with two dimensions
    # I'm using two dimensions because Rheault & Cochrane discovered meaningful dimensions with only two components
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(vectordf)

    # put PCA "ideal points" into df
    principalDf = pd.DataFrame(data=principalComponents, columns=['pc1','pc2'])

    # list of MPs in format: 'FirstName LastName_PartyName_ParliamentNo'
    # make these the indices of the df
    partylist = model.dv.index_to_key
    principalDf.index = list(partylist)

    # load statistics about speech length
    speechStats = pd.read_csv('data/speechStats.csv', header=None, names=['speechFrequency', 'totalSpeechVolume'], index_col=0)
    principalDf = principalDf.merge(speechStats, how='outer', right_index=True, left_index=True)
    principalDf.reset_index(inplace=True)
    principalDf.rename(columns={'index':'MP_party_parl'}, inplace=True)

    # separate name, party, and Parliamentary session, drop the combined version
    principalDf['speakername'] = principalDf.MP_party_parl.str.split('_').str[0]
    principalDf['speakerparty'] = principalDf.MP_party_parl.str.split('_').str[1]
    principalDf['parliamentNo'] = principalDf.MP_party_parl.str.split('_').str[2]
    principalDf.drop(['MP_party_parl'], axis=1, inplace=True)

    # drop CONGRESS from the list, as this value represents the 42nd Parliament as a whole rather than an individual MP
    allParliaments = principalDf.drop(principalDf[principalDf.speakername=='CONGRESS'].index).reset_index(drop=True)


    # Load all election results
    files = glob.glob('data/electionResults/*.csv')

    rawElections = []
    for fp in files:
        try:
            rawElections += [pd.read_csv(fp)]
        except:
            rawElections += [pd.read_csv(fp, encoding='latin-1')]

    resultsByPartyList = [generateElectionDataforRidings(rawElections[i], 38+i) for i in range(len(rawElections))]
    resultsByParty = pd.concat(resultsByPartyList, ignore_index=True)

    # add new columns to dataframe
    newColumns = ['riding', 'province', 'Liberal_share', 'Conservative_share', 'NDP_share', 'Bloc Quebecois_share', 'Green_share', 'Other_share', 'competitiveness']
    for label in newColumns:
        allParliaments[label] = [0]*len(allParliaments)

    # must make some exceptions for MPs with particular names
    # these were manually validated
    govtSupportMissingNames = ['Thomas', 'Bradley', 'Bob', 'Joe', 'Patricia', 'Michael']

    for i, MP in enumerate(allParliaments.speakername):
        for j, MpParty in enumerate(resultsByParty.MP):
            # if the MP's first and last names are in the annoyingly formatted MP label and their Parliament numbers match
            if (MP.partition(' ')[2] in MpParty) and (MP.partition(' ')[0] in MpParty) and (allParliaments.parliamentNo[i]==str(resultsByParty.ParliamentNo[j])): 
                for label in newColumns:
                    allParliaments.loc[i,label] = resultsByParty[label][j]

            # handle the name exceptions
            elif (MP.partition(' ')[2] in MpParty) and (MP.partition(' ')[0] in govtSupportMissingNames) and not ('Cathy' in MpParty) and (allParliaments.parliamentNo[i]==str(resultsByParty.ParliamentNo[j])): 
                for label in newColumns:
                    allParliaments.loc[i,label] = resultsByParty[label][j]
                

    # drop MPs who were elected in by-elections, denoted by those whose ridings had no Liberal voters
    allParliaments.drop(allParliaments[allParliaments.Liberal_share == 0].index, inplace=True)

    # add proxy measures of riding ideology based on goldstandard party ideologies
    # load goldstandard scores
    gold = pd.read_csv('data\modified_goldstandard_canada.csv')

    # function to generate score given a row of a dataframe and the name of the score type, i.e., rile, vanilla, or legacy
    def add_Score(MP, scoreName):
        """ Generate score given a row of a dataframe and the name of the score type, i.e., rile, vanilla, or legacy. """
        index_start = int(MP['parliamentNo'])-38
        denominator =100-MP['Green_share']-MP['Other_share']
        score = (gold[scoreName][index_start]*MP['Bloc Quebecois_share'] + gold[scoreName][index_start+5]*MP['Conservative_share'] + gold[scoreName][index_start+10]*MP['Liberal_share'] + gold[scoreName][index_start+15]*MP['NDP_share'])/denominator
        return score

    # add columns to the dataframes
    for tool in ['rile', 'vanilla', 'legacy']:
        allParliaments[f'{tool}Score'] = allParliaments.apply(lambda row: add_Score(row, tool), axis=1)

    # rename columns
    allParliaments.rename(columns={'pc1':'quebecker', 'pc2':'govtSupport'}, inplace=True)

    # make higher government support positive
    allParliaments.loc[:,'govtSupport'] = allParliaments['govtSupport'].apply(lambda x: x*-1)

    allParliaments.to_csv('allParliaments_joined.csv')


if __name__ == '__main__':
    main()