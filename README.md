# GPGPU

"2022_03_2022_04.ipynb" 는 lstsq를 하기 전 convolution 순전파 계산을 빨리 해보려고 이것 저것 시도해본 것입니다.<br>
"2022_04_22.ipynb" 부터는 lstsq를 하게 된 후로 공부하거나 코드 이것 저것 만들어 본 과정 결과 있는 노트입니다.<br>
밑에 있는 python 파일들을 위에서 만든 함수들을 분류해서 모아둔 파일들 입니다.<br>
개인적으로 "2022_03_2022_04.ipynb" 는 난잡해서 "Convolution_Functions.py"를 보시는 걸 추천드립니다.

Files from "lstsq/2022_05_03.ipynb" are about PyCUDA...


## to use

type 

> git clone https://github.com/Daegeun02/GPGPU

in your terminal or just download folder...

in python...

1. Least Square Solver

> from solver import *<br>
> 
> lstsq = LeastSquare(INPUTMATRIX, OUTPUTVECTOR, learning_rate, beta=beta, optimize_method="GD")<br>
> 
> lstsq.solve()<br>
> 
> optimal_theta = lstsq.shared.GPU_theta.get()

---

2. Minimum Energy Control Solver

> from solver import *<br>
> 
> MECS = MinimumEnergyControlSolver(InitialPoint, TargetPoint, UpperBoundary, DownerBoundary, Lambdas, dt, StepSize, MaxEpoch, MaxIteration)<br>
> 
> opt_u = MECS.solve()[0]
>
>detail is in code...

end...

## update teasor
MECS with changable step size...
...
ㅎ

