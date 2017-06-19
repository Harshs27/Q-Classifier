function [V] = gen_random_vector(n, a, b)
    % this function generates a vector of length n with values randomly
    % distributed from 'a' to 'b'
    V = zeros(n,1);
    for i=1:1:n,
        V(i) = a + (b-a).*rand();
    end
end