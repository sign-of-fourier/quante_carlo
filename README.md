# quante_carlo
## Batch Hyperparameter Tuning
- Multiprocessing and multi-GPU support
- Great for figuring out Neural Network Architectures
- Uses <a href="https://hal.science/hal-00732512v2/document">Bayesian Optimization Algorithm</a>
quante.carlo(f, limits, kernel, n_batches, n_processors, n_iterations, keep_thread_id=False)
<hr>

### parameters
<table>
   <tr>
      <td>  __f__ </td><td>evaluation function</td>
   </tr>
   <tr>
      <td>limits</td><td>list of parameter ranges</td>
   </tr>
   <tr>
      <td>kernel </td><td> kernel functions for Gaussian Process Regressor from sklearn.gaussian_process.kernels</td>
   </tr>
   <tr>
      <td>
- <b>n_batches</b>  number of batches to use when
- __n_processors__ number of processors; should align with actual hardware
- __n_iterations__ number of iterations to run; can alternatively specify a stopping criteria
    __logfile_location__
      </td>
      <td>
         this file will be overwritten
      </td>
   </tr>
</table>





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
