# quante_carlo
## Batch Hyperparameter Tuning
- Multiprocessing and multi-GPU support
- Great for figuring out Neural Network Architectures
- Uses <a href="https://hal.science/hal-00732512v2/document">Bayesian Optimization Algorithm</a>
<br>quante_carlo.session(f, limits, kernel, n_batches, n_processors, n_iterations, keep_thread_id=False)
<hr>

### Parameters
<table>
   <tr>
      <td><ins><v><font size="+2">Parameter Name</font></b></ins></b></td><td><ins><b><font size="+2">Description</font></b></ins></td>
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
<hr>

### Methods
<table>
   <tr>
     <td>tune(pool)</td><td>tune the evaluation function</td>
   </tr>
   <tr>
      <td>summary()</td><td>return a summary of the results</td>
   </tr>
</table>

An evaluation function must take exactly one argument (because it's getting mult-threaded using map) and it must return the score to be maximized.


## Tutorial
<img src='setup.png'><br>
Training is handled according to user specifications. Examples are available in the tutorials. Upon creating a session, quante carlo performs the evaluation in parallel and then performs the Bayesian Optimization.
quante_carlo uses Bayesian Optimization as a service. The process of obtaining optimal batch expected improvement is hadled through an API call. 


<ol>
  <li>A multi-threaded pool is created using the multiprocessing module to be used as a parallel job manager.</li>
  <li>The job manager first chooses the initial hyperparameters randomly.</li>
  <li>The job manager uses the user defined modules to make calls to the GPU. In these examples, Each instance of the module has it's own instance of a data loader. This approach scales across multiple machines and it allows each job to have a different set of hyper-parameters. The result of this step is an error or loss score for each trained model.</li>
  <li>The scores are passed to the hyperparameter microservice via API which returns the best next choice for hyperparameter tuning.</li>
</ol>
Steps 3 and 4 are repeated. Each time, the history is passed to the hyperparameter tuning service. The hyperparameter tuner transforms the data into a lognormal distribution. (Batch) expected improvement works better when the data is lognormal. The initial distribution is irrelevant. Some variables such as the betas in the Adam optimizer are usually tested on a log scale. After the transformation, the hyperparameter tuner can easily handle this skew. However, if you want to focus on an exponential distributed parameter, then you can handle that in the user-defined function, for example, by passing the log of the parameter to the hp tuner.
<hr>
<h2>Notes </h2>
<ul>
  <li>The parameters are transformed into lognormals by using the probability integral transform. This is a better way to scale, especiaily for Bayesian Optimization</li>
  <li>Expected Improvement of a variable is caluclated by converting E[max(x) | <i>x<sub>i</sub></i> > best] for all <i>i</i> by defining new variables <i>z<sub>i</sub></i> = <i>x<sub>i</sub> - x<sub>j</sub></i> and calculating E[<i>x<sub>i</sub></i> | <i>x<sub>i</sub></i> > best & <i>x<sub>i</sub></i> > x~j] for all <i>i</i> &ne; <i>j</i></li>
</ul>
 
