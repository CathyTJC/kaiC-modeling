KaiC_func <- function(t, X, params) {
  C_kaiC <- 3.4
  C_kaiA <- 1.3
  m <- 1
  K_half <- 0.43
  A = max(0, C_kaiA-2*m*X[,3])
  #c(Kut,Ktd,Ksd,Kus,Ktu,Kdt,Kds,Ksu)
  K_tota =
    c(0,0,0,0,0.21,0,params[1], params[2]) +
    (c(0.479077,0.212923,0.505692,0.0532308,0.0798462,0.1730000,-0.319385,-0.133077)*A) /(K_half + A)
  U = C_kaiC -(X[,1]+X[,2]+X[,3])
  dxdt <- cbind(
    K_tota[1] * U + K_tota[6] * X[,2] - K_tota[5] * X[,1] - K_tota[2] * X[,1], 
    K_tota[2] * X[,1] + K_tota[3] * X[,3] - K_tota[6] * X[,2] - K_tota[7] * X[,2],
    K_tota[4] * U + K_tota[7] * X[,2] -K_tota[8] * X[,3] - K_tota[3] * X[,3]
  )
  return(dxdt)
}

library('deSolve')
# Wrapper function
deSolve_kaiC_func = function(t,y,params) {
  list(KaiC_func(t,matrix(y,1,length(y)),params))
}
# Generate some data
test.times = seq(0,100,0.1)
test.data = ode(c(0.68,1.36,0.34), test.times, deSolve_kaiC_func, c(0.31,0.11))
plot(test.data)
test.data = test.data[,2:4]


library('deGradInfer')
#run adaptive gradient matching
agm.result = agm(test.data,test.times,ode.system = KaiC_func, numberOfParameters=2,maxIterations=500,chainNum=5)
print(agm.result$posterior.mean)
#true value: K0 =[0,0,0,0,0.21,0,0.31,0.11]
#true value: KA =[0.479077,0.212923,0.505692,0.0532308,0.0798462,0.1730000,-0.319385,-0.133077]

#result for 2 parameters
# 1.752164 3.192209

# result for the last 2 of K0
# 2.773095 1.958643

library('ggplot2')
plotting.frame = data.frame(Param.Values=c(agm.result$posterior.samples),
                            Param.Num=rep(1:2, each=dim(agm.result$posterior.samples)[1]))
ggplot(plotting.frame, aes(x=Param.Values, y=..scaled..)) +
  facet_wrap(~paste('Parameter', Param.Num), scales = 'free') +
  geom_density() +
  labs(y='Scaled Density', x='Parameter Value') +
  xlim(c(0,5))
