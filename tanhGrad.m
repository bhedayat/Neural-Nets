function [ output_args ] = tanhGrad( input_args )


output_args = 1.14393*(1 - (tanh((2/3)*input_args)).^2);

end

