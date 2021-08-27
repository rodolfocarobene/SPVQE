import functools
import numpy as np

def ei(i,n):
    vi = np.zeros(n)
    vi[i] = 1.0
    return vi[:]

class manual_VQE:
      def __init__(self,H,Hoff,A,Aoff,generate_circuit,instance,num_parameters):
          self.H    = H
          self.Hoff = Hoff
          self.A    = A
          self.Aoff = Aoff
          self.num_parameter    = num_parameters
          self.generate_circuit = generate_circuit
          self.instance         = instance

      def read_from_file(self,fname):
          f     = open(fname,'r')
          lines = f.readlines()
          s0,i0 = lines[0].split()
          s1,i1 = lines[1].split()
          i0,i1 = int(i0),int(i1)
          parameter = np.zeros(i1)
          for m,l in enumerate(lines[len(lines)-i1:]):
              parameter[m] = l.split()[1]
          return i0+1,i1,parameter

      def measure(self,operator,wfn_circuits,instance):
          circuits = []
          for idx,wfn_circuit in enumerate(wfn_circuits):
              circuit = operator.construct_evaluation_circuit(
                        wave_function               = wfn_circuit,
                        statevector_mode            = instance.is_statevector,
                        use_simulator_snapshot_mode = False,
                        circuit_name_prefix         = 'wfn_'+str(idx))
              circuits.append(circuit)
          # ---
          if circuits:
              to_be_simulated_circuits = \
                  functools.reduce(lambda x, y: x + y, [c for c in circuits if c is not None])
              result = instance.execute(to_be_simulated_circuits)
          # ---
          results_list = []
          for idx,wfn_circuit in enumerate(wfn_circuits):
              mean,std = operator.evaluate_with_result(
                         result = result,statevector_mode = instance.is_statevector,
                         use_simulator_snapshot_mode = False,
                         circuit_name_prefix         = 'wfn_'+str(idx))
              mean,std = np.real(mean),np.abs(std)
              results_list.append([mean,std])
          # ---
          return results_list

      def measure_ancillas(self,parameters):
          circuits = []
          wfn_circuits = [self.generate_circuit(parameters)]
          for jdx,oper in enumerate(self.A):
              if(not oper.is_empty()):
                 for idx,wfn_circuit in enumerate(wfn_circuits):
                     circuit = oper.construct_evaluation_circuit(
                               wave_function               = wfn_circuit,
                               statevector_mode            = self.instance.is_statevector,
                               use_simulator_snapshot_mode = False,
                               circuit_name_prefix         = 'oper_'+str(jdx)+'_wfn_'+str(idx))
                     circuits.append(circuit)
          # ---
          if circuits:
              to_be_simulated_circuits = \
                  functools.reduce(lambda x, y: x + y, [c for c in circuits if c is not None])
              result = self.instance.execute(to_be_simulated_circuits)
          # ---
          results_list = []
          for jdx,oper in enumerate(self.A):
              if(not oper.is_empty()):
                 for idx,wfn_circuit in enumerate(wfn_circuits):
                     mean,std = oper.evaluate_with_result(
                                result = result,statevector_mode = self.instance.is_statevector,
                                use_simulator_snapshot_mode = False,
                                circuit_name_prefix         = 'oper_'+str(jdx)+'_wfn_'+str(idx))
                     mean,std = np.real(mean),np.abs(std)
                     results_list.append([mean,std])
              else:
                 results_list.append([0,0])
          # ---
          return results_list

      def evaluate_energy_and_gradient(self,parameters):
          # in un solo circuito valutare E e gradiente di E
          nparameters  = len(parameters)
          wfn_circuits = [self.generate_circuit(parameters)]
          for i in range(nparameters):
              wfn_circuits.append(self.generate_circuit(parameters+ei(i,nparameters)*np.pi/2.0))
              wfn_circuits.append(self.generate_circuit(parameters-ei(i,nparameters)*np.pi/2.0))
          results = self.measure(self.H,wfn_circuits,self.instance)
          E = np.zeros(2)
          g = np.zeros((nparameters,2))
          E[0],E[1] = results[0]
          for i in range(nparameters):
              rplus  = results[1+2*i]
              rminus = results[2+2*i]
              # G      = (Ep - Em)/2
              # var(G) = var(Ep) * (dG/dEp)**2 + var(Em) * (dG/dEm)**2
              g[i,:] = (rplus[0]-rminus[0])/2.0,np.sqrt(rplus[1]**2+rminus[1]**2)/2.0
          return E,g

      def perform_line_search(self,parameters,gradient,t_mesh): #perator,c0,parameters,gradient,t_mesh):
          npoints = len(t_mesh)
          wfn_circuits = []
          for t in t_mesh:
              wfn_circuits.append(self.generate_circuit(parameters-gradient*t))
          E_mesh = self.measure(self.H,wfn_circuits,self.instance)
          t_opt,E_opt = None,None
          from scipy.optimize import curve_fit
          def f(x,a0,k,x0):
              return a0+(k/2.0)*(x-x0)**2
          E_average = [E[0] for E in E_mesh]
          E_stdev   = [E[1] for E in E_mesh]
          p0 = [min(E_average),1.0,t_mesh[np.argmin(E_average)]]
          p_opt,p_cov = curve_fit(f,t_mesh,E_average,sigma=E_stdev,absolute_sigma=True,method='lm')
          t_opt = [p_opt[2],np.sqrt(p_cov[2,2])]
          E_opt = [p_opt[0],np.sqrt(p_cov[0,0])]
          # E(t) = E(x0-t*g) ~ a* x^2 + bx + c
          return t_mesh,E_mesh,t_opt,E_opt

      def dump_on_file(self,iteration_counter,parameter,A_ave,E,gradient,t_mesh,E_mesh,t_opt,E_opt,dE):
          f = open('vqe_'+str(iteration_counter)+'.txt','w')
      
          f.write('iteration:  %d\n' % (iteration_counter))
          f.write('parameters: %d\n' % (len(parameter)))
          for i,p in enumerate(parameter):
              f.write('%d %f\n' % (i,p))
      
          f.write('ancillas\n')
          for i,(m,s) in enumerate(A_ave):
              f.write('%d %f +/- %f\n' % (i,m,s))

          f.write('energy: = %f +/- %f\n' % (E[0]+dE,E[1]))
          f.write('gradient: \n')
          for i,(mg,sg) in enumerate(gradient):
              f.write('%d %f +/- %f\n' % (i,mg,sg))
     
          f.write('line search: \n')
          for t,(em,es) in zip(t_mesh,E_mesh):
              f.write('%d %f +/- %f\n' % (t,em+dE,es))
      
          f.write('proposed new energy: %f +/- %f\n' % (dE+E_opt[0],E_opt[1]))
          f.write('proposed parameters:\n')
          parameter = parameter-t_opt[0]*np.array([g[0] for g in gradient])
          for i,p in enumerate(parameter):
              f.write('%d %f\n' % (i,p))
      
          import matplotlib.pyplot as plt
          #fig,ax = plt.subplots(1,2)
          #plt.subplots_adjust(hspace=0.3,wspace=0.3)
          L = 7
          fig,ax = plt.subplots(1,2,figsize=(2*L,0.66*L))

          ax[0].errorbar(range(len(gradient)),[g[0] for g in gradient],yerr=[g[1] for g in gradient])
          ax[1].errorbar(t_mesh,[dE+E[0] for E in E_mesh],yerr=[E[1] for E in E_mesh])
          ax[0].set_xlabel('component')
          ax[0].set_ylabel('gradient [Ha]')
          ax[1].set_xlabel('t [1/Ha]')
          ax[1].set_ylabel('energy [Ha]')
          plt.savefig('vqe_step_'+str(iteration_counter)+'.eps')

      def run(self,t_mesh):

          from os import path
          starting_iteration = not path.exists("vqe_0.txt")
          
          if(starting_iteration):
             iteration_counter = 0
             parameters        = np.zeros(self.num_parameter)
          else:
             import glob
             f_list = glob.glob("vqe_*.txt")
             i_list = [int(f[4:len(f)-4]) for f in f_list]
             iteration_counter,nparameters,parameters = self.read_from_file('vqe_'+str(max(i_list))+'.txt')

          A_ave = self.measure_ancillas(parameters)
          E,g = self.evaluate_energy_and_gradient(parameters)
          t_mesh,E_mesh,t_opt,E_opt = self.perform_line_search(parameters,g[:,0],t_mesh)
          self.dump_on_file(iteration_counter,parameters,A_ave,E,g,t_mesh,E_mesh,t_opt,E_opt,self.Hoff)

