# Data-driven Mixed Precision Sparse Matrix Vector Multiplication for GPUs (MpSpMV)

In the publication linked below we used C code to split sparse matrices. However for ease of use this repository hosts a Matlab version of the split code.

Related publication available here: https://dl.acm.org/doi/pdf/10.1145/3371275

The following instructions assume that the sparse matrix is not symmetrical. If the sparse matrix you want to split into two regions please expand the symmetric matrix and then use these instructions.


```
Temp = load('cop20k_A.mtx');
in = find(Temp(:,3)<=2);
out = find(Temp(:,3)>2);
mat_in = Temp(in,:);
mat_out = Temp(out,:);

mmwrite('cop20k_A_in.mtx', spconvert(mat_in))
mmwrite('cop20k_A_out.mtx', spconvert(mat_out))

%%Visualize the orginal sparse matrix
spy(spconvert(Temp))
%%Visualize the non zero values that are **inside** the range -2 to 2 from the orginal sparse matrix
figure
spy(spconvert(mat_in))
%%Visualize the non zero values that are **outside** the range -2 to 2 from the orginal sparse matrix
figure
spy(spconvert(mat_out))
```

![Orginal Sparse Matrix](https://github.com/KhalidTheeb/MpSpMV/blob/master/Img/Org.jpg "Orginal Sparse Matrix")
![Inner Values](https://github.com/KhalidTheeb/MpSpMV/blob/master/Img/In.jpg)
![Outter Values](https://github.com/KhalidTheeb/MpSpMV/blob/master/Img/Out.jpg)
