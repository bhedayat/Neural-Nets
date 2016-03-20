function [ output_args ] = softmax(input_args)


% output_args=bsxfun(@rdivide, exp(input_args), sum(exp(input_args),2));

num_labels = size(input_args,2);
for c = 1:num_labels
    output_args(:,c) = exp(input_args(:,c))./sum(exp(input_args),2);       
end

end

