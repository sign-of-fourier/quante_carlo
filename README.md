# quante_carlo
## Batch Hyperparameter Tuning
- Multiprocessing and multi-GPU support
- Great for figuring out Neural Network Architectures
- Uses <a href="https://hal.science/hal-00732512v2/document">Bayesian Optimization Algorithm</a>
quante.carlo(f, limits, kernel, n_batches, n_processors, n_iterations, keep_thread_id=False)
<hr>

### Parameters
<table>
   <tr>
      <td><u><b><font size="+2">Parameter Name</font></b></u></b></td><td><u><b><font size="+2">Description</font></b></u></td>
   </tr>
   <tr>
      <td><b>f<b> </td> </td><td>evaluation function</td>
   </tr>
   <tr>
      <td><b>limits</b></td><td>list of parameter ranges</td>
   </tr>
   <tr>
      <td><b>kernel</b> </td><td> kernel functions for Gaussian Process Regressor from sklearn.gaussian_process.kernels</td>
   </tr>
   <tr>
      <td><b>n_batches</b></td><td>number of batches to use when</td>
   </tr>
   <tr>
         <td><b>n_processors</b></td><td> number of processors; should align with actual hardware</td>
   </tr>
   <tr>
      <td><b>n_iterations</b></td><td> number of iterations to run; can alternatively specify a stopping criteria</td>
   </tr>
         <td><b>logfile_location</b></td> <td> this file will be overwritten </td>
   </tr>
</table>





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
