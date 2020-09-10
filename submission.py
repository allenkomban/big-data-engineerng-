## import modules here



########## Question 1 ##########
# do not change the heading of the function
def c2lsh(data_hashes, query_hashes, alpha_m, beta_n):

    y=data_hashes.map(lambda x: (x[0],[abs(a_i - b_i) for a_i, b_i in zip(x[1], query_hashes)]))


    offset=0



    while True:

        h=y.filter(lambda x : len([ y for y in x[1] if y<=offset ]) >=alpha_m )

        if(h.count()>=beta_n):
            #f=h.map(lambda x:x[0])
            f=h.keys()
            break

        else:
            offset=offset+1

    return f


