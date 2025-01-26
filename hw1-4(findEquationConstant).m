
% Given values (replace these with your actual values)
tx = [128; 124; 113; 111]; % Output values
gx = [128; 144; 128; 144]; % gx values
gy = [128; 128; 144; 144]; % gy values

% Construct the matrix for the system of equations
A = [gx, gy, gx .* gy, ones(4, 1)];
% Solve for coefficients a, b, c, d
disp('The value of matrix A is:');
disp(A);
coefficients = A \ tx;

% Extract coefficients
a = coefficients(1);
b = coefficients(2);
c = coefficients(3);
d = coefficients(4);

% Display the results
fprintf('a = %.4f\nb = %.4f\nc = %.4f\nd = %.4f\n', a, b, c, d);
% Calculate the determinant
d = det(A);

% Display the result
disp('Determinant of the matrix is:');
disp(d);