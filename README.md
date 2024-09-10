# Relative-Entropy
This code calculates the relative entropies between prior distributions and the posterior distributions obtained by BXA.

The code's current form was used to calculate the relative entropies in Figure 6 of this paper (link paper). 

Only uniform and loguniform prior distributions are included.

## To use:
The important functions are the relative_entropy(posterior,a,b) and log_relative_entropy(posterior,a,b). It is recommended to simply copy these functions to use in your own code.

If your workflow involves multiple sources and the models borus02 and UXCLUMPY, much of the code could be useful. You would need to change the path to the posterior.csv file. Also, you would need to change the names of the parameters and the bounds in the prior_borus and prior_uxclumpy dictionaries to match your specific case. 
