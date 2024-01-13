# AB-Test-Groups-KMeans

The file generate_groups.py takes in a csv file with users and n columns with numeric features, and spits out the same csv but with clusters defined by KMeans algorythm and groups A, B and Control.

**Why KMeans Clustering?**
* The clusters will help you split your users among the 3 possible group in a stratified randomization
* That means, for each cluster N, there will be N/3 equal parts in each group A, B and Control
* This can be very useful to divide users in a AB Test
* This way of generating groups is recommended when your users have too many complex features that, if simply randomized, might still not generate conclusive hypotesis testing, for example:
  * Different demographics that have extremely different behaviours
  * If your test is affected by geospatial position and you have few users for a large area (in that case you can use lat-lon as part of the model features)
  * If your users are consuming your product for very different reasons or frequencies
 
**How to run?**
* It's recommended to read all commends before running generate_groups.py
* The initial .csv file name is set to be your_csv.csv but that can be changed at line 9
* The suggested setup is to run this code in a JupyterNotebook or in an IDE that allows you to run by blocks
* It's highly suggested to add cleaning steps after line 25
* Adapt 'n_init' variable at line 43 depending on your computational availability
* You must run the code until line 59, then deciding on how many clusters to go with, and proceed after editing the variable 'num_clusters' at line 62

# Contributions

Arthur Lee - Data Scientist (2024-01-13)
