# Experiments of Development of an Approach to Automation of Foreign Enrollee Intellectual Support Based on NLP-Technologies and Information Crawling

## Table of Contents

1. Project Description
2. Technologies
3. Files Description
4. Result
5. Licensing and Authors

## Project Description
The project are for experimenting the performance of modifed NLP approaches
1. hypercomplex classifer from pre-trained RoBERT, aiming at
   - More sensitive classifer than original fully-connected classifer
   - More light-weight model sturcture, and comparative performance as original classifer does
2. GPT text augmentation with pos-tagging, aiming at
   - Optimiziing previous work from [Data Augmentation Using Pre-trained Transformer Models](https://aclanthology.org/2020.lifelongnlp-1.3.pdf)
   - By using Pos-tagging to seize kyeword and augment context
   - In previous work, author used word as label, for further development in system it is changed to numeric label
   
## Technologies
Technique : 
* Hypercomplex classifer implemented in pre-trained RoBERT model
* GPT augmentation with Pos-taggin

## Files Description
* data augmentation
   * eda.py -- easy data augmentation
   * backtranslation.py -- multiple languages translation data augmentation
   * cbert.py -- conditional bert data augmentation
   * cgpt.py -- conditional gpt data augmentation
   * cPosGpt2.py -- proposed data augmentation with postagging with gpt
   * augmentation_before_classification_stas.ipynb -- demo of augmentation from above py files
* Hypercomplex classifer
   * current_search_new.ipynb -- demo of hypercomplex classifer from pre-trained RoBERT with Trec datasets
   * current_search_gpt2.ipynb -- domo of hypercomplex classifer from pre-trained Gpt with Trec datasets
* kmean_cluster.py -- evaluation metircs, such as umap, euclidean and kl_divergence_

## Result

* Hypercomplex classifier

![alt text](https://github.com/MaChengYuan/current_work/blob/main/%E6%88%AA%E5%9C%96%202024-05-02%20%E4%B8%8A%E5%8D%881.44.14.png)

![alt text](https://github.com/MaChengYuan/current_work/blob/main/%E6%88%AA%E5%9C%96%202024-05-02%20%E4%B8%8A%E5%8D%881.47.06.png)

* Data Augmentation

![alt text](https://github.com/MaChengYuan/current_work/blob/main/%E6%88%AA%E5%9C%96%202024-05-02%20%E4%B8%8A%E5%8D%881.47.32.png)
  

## Licensing and Authors

Fernandodoro - ITMO student

Reach me at:

tommy0101@hotmail.com.tw

Licensing:[MIT LICENSE](https://github.com/MaChengYuan/current_work/blob/main/LICENSE)


