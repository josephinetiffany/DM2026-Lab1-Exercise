# Agentic Pipeline Report

## 1. Data loading and cleaning

Using `load_dataset_1`, we loaded 2,257 total documents across four topic categories: `soc.religion.christian` (599 posts), `sci.med` (594 posts), `comp.graphics` (584 posts), and `alt.atheism` (480 posts). The groups are mostly even in size, though `alt.atheism` is slightly smaller, which is helpful to keep in mind since smaller categories can look less noticeable on plots later. Checking the data with `check_missing_2` confirmed 0 missing values, and `check_duplicates_4` confirmed 0 duplicate rows, so we kept all 2,257 rows for the next steps.

## 2. Document-term matrix and term frequency

Building the document-term matrix with `build_dtm_5` created a grid of 2,257 documents by 35,788 unique words, with a sparsity of 99.547%. Because the tool kept stopwords like "the" and "is", we got 35,788 words instead of the 35,482 words from Master Q4 (where stopwords were removed). Keeping these extra 306 words slightly lowered the sparsity from 99.68% (from my Master Q4 run) down to 99.547% because common words show up in almost every row. As I predicted, 'the' and ‘of’ led the list, and 19 of the top 20 were common filler words, with the only non-stopword being "edu" from email addresses. Because of this, raw word counts do not help tell the topics apart, suggesting that removing stopwords and using TF-IDF would likely help. The bar chart was made with `visualize_result_tool` on `term_frequency_6`.

![Top 15 terms](../plots/Josephine_Tiffany_Setio_agentic-pipeline_plot_turn48.png)

## 3. PCA projection and spatial structure

Running PCA via `reduce_dimensions_7` showed that the first axis (PC1) accounts for 65.94% of the spread in the data, while the second axis (PC2) accounts for just 4.33%. As I predicted, most documents form one big mixed clump in the middle of the graph. A few `comp.graphics` posts stretch high up on PC2, while `soc.religion.christian` and `alt.atheism` posts trail off together toward the bottom right, including one far-off `alt.atheism` point out at PC1 ≈ 830. PC1 usually tracks how long a document is or how many total filler words it has, though that is an educated guess rather than something computed directly. The points sitting far outside the center are likely posts that repeat specific words over and over.

![PCA](../plots/Josephine_Tiffany_Setio_agentic-pipeline_plot_reduce_dimensions_7.png)

## 4. UMAP manifold analysis

UMAP via `reduce_dimensions_8` spread out the data much better than PCA, showing clearer groups for each topic as I predicted: `comp.graphics` sits near the bottom right, `soc.religion.christian` moves to the top left, and `alt.atheism` forms a stripe across the middle, while `sci.med` is the most scattered across the plot. However, the groups are not completely separate, so the prediction was only partly right, as `alt.atheism` and `soc.religion.christian` still overlap a lot along their shared border. UMAP works better than PCA at grouping similar posts together, but keeping stopwords probably makes UMAP cluster posts based on common grammar rather than real topic content. Setting options like `n_neighbors` or `min_df` differently could also change how clean these groups look.

![UMAP](../plots/Josephine_Tiffany_Setio_agentic-pipeline_plot_reduce_dimensions_8.png)

## 5. Conclusion and future improvements

Using raw word counts without removing common filler words makes the data dominated by useless terms, causing most documents to lump together in PCA. While UMAP does a better job showing underlying patterns and grouping related posts, filler words still blur the boundaries between topics. To get cleaner groups in future runs, we should filter out English stopwords, use TF-IDF to tone down frequent words, and adjust UMAP settings like `n_neighbors` and `min_dist`.