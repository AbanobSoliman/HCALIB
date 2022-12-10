function S = cumul_b_splineR3(P,u,N) %#codegen

    S = zeros(3,length(u)*(length(P)-N));
    
    k = N + 1;
    U = zeros(k,length(u));
    dP = get_inc(P);
    
    M = calcM(k);

    for i = 1 : k
        U(i,:) = u.^(i-1); 
    end
    
    zz = 1;
    for i = 1 : size(P,2)-k+1       
        B = [P(:,i),dP(:,i:i+k-2)]*M*U; % B-spline segment            
        S(:,zz:zz+size(B,2)-1) = B;
        zz = zz + size(B,2);          
    end

end

function dX = get_inc(X) %#codegen

    if isa(X,'double')
        
        for i = 1 : size(X,2)-1
            dX(:,i) = X(:,i+1) - X(:,i);
        end
        
        return
        
    elseif isa(X,'quaternion')
        
        for i = 1 : length(X)-1
            dX(i,:) = quatmultiply( compact(X(i)) , quatinv(compact(X(i+1))) );
        end
        
        dX = dX';
        
        return
        
    else
        
        disp('This function gets increments of positions(doubles) and quaternions!');
        return 
        
    end
    
end

function M = calcM(k) %#codegen

    M = zeros(k);
    m = zeros(k);
    
    s = 0:1:k-1;
    n = 0:1:k-1;
    L = 0:1:k-1;
    for i = 1 : k % column
        for j = 1 : k % row
            add = 0.0;
            for l = i : k
                add = add + double((-1)^(L(l)-s(i))*nchoosek(k,L(l)-s(i))*(k-1-L(l))^(k-1-n(j)));
            end
            m(i,j) = (nchoosek(k-1,n(j))/(factorial(k-1)))*add; % Basis function matrix
        end
    end

    for j = 1 : k % column
        for n = 1 : k % row
            M(j,n) = sum(m(j:k,n)); % Cumulative Basis function matrix
        end
    end

end

