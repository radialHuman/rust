## Description
> To make a library of functions that are frequently used for data anlaysis and machine learning tasks

## Changes
* lib_nn the functions and Struct are tested, modified and documented

## Planned
1. lib_string


## List of Functions and Structs

### lib_matrix
    1. MatrixDeterminantF : 
        > determinant_f
            x determinant_2
            x determinant_3plus
        > is_square_matrix
        > round_off_f
        > inverse_f
            x identity_matrix
            x zero_matrix

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
---
### lib_ml
    1. MultivariantLinearRegression :
        > multivariant_linear_regression
            x generate_score
        > batch_gradient_descent
            x mse_cost_function
        > hash_to_table
            x train_test_split
            x randomize
            
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
    15. randomize
    16. train_test_split
    17. correlation
    18. std_dev
    19. s_rank
    20. how_many_and_where
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
    
----
### About the author
* Used Python, learning Rust
* Not a CS student, feedback appreciated
* rd2575691@gmail.com
---
### Vibliography ?
* For lib_nn : [nnfs.io](https://www.youtube.com/playlist?list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3)
* For Rust : [Crazcalm's Tech Stack](https://www.youtube.com/playlist?list=PLVhhUNGAUIQScqB26DdUq4n1Y2n3auM7X)
* Other blogs mentioned inside
