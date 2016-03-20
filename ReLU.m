function [ output_args ] = ReLU( input_args )


[m,n] = size(input_args);
output_args = zeros(m,n);
output_args(input_args>0) = input_args(input_args > 0);


end

