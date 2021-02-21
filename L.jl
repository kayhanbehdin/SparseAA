###############
# L(H_1, H_2) #
###############
function L(H1, H2)
    d = 0
    k, n = size(H1)
    for i = 1:k
        dmin = 1e20
        for j = 1:k
            temp = norm(H1[i,:] - H2[j,:],2)^2
            if (temp < dmin)
                dmin = temp
            end
        end
        d = d + dmin
    end
    return d
end
