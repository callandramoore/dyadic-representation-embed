# dyadic-representation-embed
Generate, analyze, and plot Canadian MP embeddings versus election results to analyze dyadic representation. 

## Tour of the repository
* `preprocessing.py`: functions to remove uninformative speeches and speakers, clean (stemming, remove accents, remove punctuation, etc.) and tokenize text, and format for use in `partyembed`. Takes raw data from [lipad.ca](www.lipad.ca) to {parliament number}Parl.csv files to 38to42Parl_forembeddings.csv. 
* `speech_lengths.ipynb`: generates histograms speech length distributions; contains analysis of low speech lengths to determine speech length cutoff. 
* `speech_stats.py`: generate speech frequency and total speech volume for each MP in each Parliament. Takes 38to42Parl_forembeddings.csv to speechStats.csv. 
* `partyembed`: submodule forked from [`lrheault/partyembed`](https://github.com/lrheault/partyembed) with some updates for MP (rather than party) embeddings, plotting utilities, and gensim updates [1]. 
    * `partyembed/explore.py`: load and plot Doc2Vec model
    * `partyembed/utils/interpret.py`: get words associated with each principal component pole
    * `src/partyembeddings_house.py`: generate MP embeddings using Doc2Vec. Takes 38to42Parl_forembeddings.csv to 38to42Parl model.  
* `join_data.py`: join principal component values, election results, speech volume and frequency statitics, and MP/riding metadata into a single csv. Takes 38to42Parl model, speechStats.csv, electionResults/*.csv, modified_golstandard_canada.csv to allParliaments_joined.csv. 
* `plots.ipynb`: generate plots comparing principal component values and election outcomes using utils in `plot_helpers.py` and allParliaments_joined.csv. 

## Usage
1. Clone repository: `git clone --recurse-submodules https://github.com/callandramoore/dyadic-representation-embed.git`

2. Install requirements: `pip install -r requirements.txt`

3. Unzip data, move model to `partyembed/partyembed/models/`. 


## Citations
[1] Rheault, Ludovic, and Christopher Cochrane. 2020. “Word Embeddings for the Analysis of Ideological Placement in Parliamentary Corpora.” Political Analysis 28 (1). Cambridge University Press: 112–33. doi:10.1017/pan.2019.26. Repository available: [`github.com/lrheault/partyembed`](https://github.com/lrheault/partyembed)