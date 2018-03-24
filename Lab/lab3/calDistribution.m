function [mean_val, std_val] = calDistribution(in_data, var_name)
mean_val = mean(in_data.time);
std_val = std(in_data.time);

figure()
hist(in_data.time, 5)
title([var_name, '=', num2str(in_data.var),...
    ' (mean:', num2str(mean_val), ...
    ', SD: ', num2str(std_val), ')'])
grid on 
end