clear all;
n = 300;
N = 24;
x = rand(1,n);
v = -0.1 + (0.1 - (-0.1)).*rand(1,n);
d = sin(20*x) + 3 * x + v;
figure(3)
scatter(x,d);
b_2 = zeros(N,1);
b_2(1) = rand(1);
w_1 = rand(N,1);
b_1 = rand(N,1);
w_2 = rand(N,1);
assistS2 = zeros(1 , 1);
assistS1 = zeros(N , N);
eta = 0.01;
final_output = zeros(n,1);
e = zeros(1,n);
count = 1;
minError = 0.1;
sumMSE1 = 1;
sumMSE2 = 1;
change = zeros(100,1);
j = 1;
sumMSE1 = zeros(1000000,1);
sumMSE1(1) = 1;

while(sumMSE1(count) >= minError )
   count=count+1;
for i = 1:n
    middle_input = w_1 * x(1,i)+b_1;
    middle_output = tanh(middle_input);
    final_input = middle_output' * w_2;
    final_output(i,1) = final_input;
    assistS1 = diag(1 - (middle_output .* middle_output));
    assistS2 = diag(ones(size(final_output(i,1))));
    e = d(i) - final_output(i);
    s2 = -2 * assistS2 * e; 
    s1 = assistS1 * w_2 * s2;
    w_2 = w_2 - eta * s2 * middle_output; 
    b_2(1) = b_2(1) - eta * s2; 
    w_1 = w_1 - eta * s1 * x(i); 
    b_1 = b_1 - eta * s1;
    sumMSE1(count) = sumMSE1(count) + e * e;
   
     
end
    sumMSE1(count) = sumMSE1(count) / n;
    if (count>100 && sumMSE2<sumMSE1(count))
       eta = eta*0.9;
 
   change(j) = count;
   j = j + 1;
    end
    
    sumMSE2 = sumMSE1(count);
    
end
figure(1)
plot(1:count,sumMSE1(1:count,1));
W = [w_1,b_1,w_2,b_2];


for i = 1:n
  middle_input = w_1 * x(1,i)+b_1;
    middle_output = tanh(middle_input);
    final_input = middle_output' * w_2;
    final_output(i,1) = final_input;
end
figure(2)
scatter(x , final_output , '.');
hold on;
scatter(x , d , 'v');
hold off;