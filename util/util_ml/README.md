## Description
> To make a library of functions that are frequently used for data anlaysis and machine learning tasks

## Changes
* lib_ml : outlier finder, Percentile finder, quartile calculator
* lib_matrix : groupby (sum and mean)

## List of Functions and Structs

### lib_matrix
    1. MatrixDeterminantF : 
        > determinant_f
            x determinant_2
            x determinant_3plus
        > is_square_matrix
            x round_off_f
        > inverse_f
            x identity_matrix
            x zero_matrix

    2. DataFrame:
        > groupby

    1. dot_product
    2. element_wise_operation
    3. matrix_multiplication
    4. pad_with_zero
    5. print_a_matrix
    6. shape_changer
    7. transpose
    8. vector_addition
    9. make_matrix_float
    10. make_vector_float
    11. round_off_f
    12. unique_values
    13. value_counts
    14. is_numerical
    15. min_max_f
    16. type_of
    17. element_wise_matrix_operation
    18. matrix_vector_product_f
    19. split_vector
    20. split_vector_at
    21. join_matrix
    22. make_matrix_string_literal
    23. head
    24. tail
    25. row_to_columns_conversion
    26. columns_to_rows_conversion

---
### lib_ml
    1. OLS:
        > fit
    2. BLR:
        > fit
        > sigmoid
        > log_loss
        > gradient_descent
        > change_in_loss
        > predict
    3. KNN
        > fit
        x predict
    4. Distance
        > distance_euclidean
        > distance_manhattan
        > distance_cosine
        > distance_chebyshev
    5. Kmeans
        > fit

    1. coefficient
    2. convert_and_impute
    3. covariance
    4. impute_string
    5. mean
    6. read_csv
    7. root_mean_square
    8. simple_linear_regression_prediction
    9. variance
    10. convert_string_categorical 
    11. normalize_vector_f
    12. logistic_function_f
    13. log_gradient_f 
    14. logistic_predict 
    15. randomize_vector
    16. randomize
    17. train_test_split_vector_f
    18. train_test_split_f
    19. correlation
    20. std_dev
    21. spearman_rank
    22. how_many_and_where_vector
    23. how_many_and_where
    24. z_score
    25. one_hot_encoding
    26. shape
    27. rmse
    28. mse
    29. mae
    30. r_square
    31. mape
    32. drop_column
    33. preprocess_train_test_split
    34. standardize_vector_f
    35. min_max_scaler
    36. float_randomize
    37. confuse_me
    38. cv
    39. z_outlier_f
    40. percentile_f
    41. quartile_f
---
### lib_nn
    1. LayerDetails :
        > create_weights
        > create_bias
        > output_of_layer

    1. activation_leaky_relu
    2. activation_relu
    3. activation_sigmoid
    4. activation_tanh
---
### lib_string
    1. StringToMatch :
        > compare_percentage
            x calculate
        > clean_string
            x char_vector
        > compare_chars
        > compare_position
        > fuzzy_subset
            x n_gram
        > split_alpha_numericals
        > char_count
        > frequent_char
        > char_replace
    
    1. extract_vowels_consonants
    2. sentence_case
    3. remove_stop_words
    
---
### Comparision with [Scikit learn's output](https://github.com/radialHuman/rust_ml/tree/master/from_scratch/src)
* OLS
* BLR
* KNN
* Kmeans
----
### About the author
* Used Python, learning Rust
* Feedback appreciated
* rd2575691@gmail.com
---
### Vibliography ?
* For Rust : [Crazcalm's Tech Stack](https://www.youtube.com/playlist?list=PLVhhUNGAUIQScqB26DdUq4n1Y2n3auM7X)
* For lib_nn : [nnfs.io](https://www.youtube.com/playlist?list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3)
* Other blogs mentioned inside
