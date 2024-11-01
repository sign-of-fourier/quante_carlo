
class rapids_batch_rf(base_worker):
    
    def __init__(self, max_depth, min_samples_leaf, min_samples_split, impurity_decrease):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.impurity_decrease = impurity_decrease
        self.all_predictions = []
        self.truth = []
        super().__init__(gbr_batch_size, n_processors, 'rapids')

    def get_file(self, path, f):
        if f == 'xaa':
            nvidia_df = pd.read_csv(path + '/' + f, nrows=100)
        else:
            headers = pd.read_csv(path + '/xaa', nrows=1)
            nvidia_df = pd.read_csv(path + '/' + f, nrows=100, header=None)
            nvidia_df.columns = headers.columns
            
        float_cols = [c for c in nvidia_df if nvidia_df[c].dtype == "float64"]
        float32_cols = {c: np.float32 for c in float_cols}
        if f == 'xaa':
            nvidia_df = pd.read_csv(path + '/' + f, dtype=float32_cols)
        else:
            nvidia_df = pd.read_csv(path + '/' + f, dtype=float32_cols, header=None)
            nvidia_df.columns = headers.columns
        return nvidia_df
        
    def prep(self, df, enc=None, kmodel=None):
            
        continuous_predictors = df.columns[4:]
        categorical = ['trickortreat', 'kingofhalloween']
        if not enc:
            enc = OneHotEncoder(handle_unknown='ignore')
            enc = enc.fit(df[categorical])
            
        categorical_df = enc.transform(df[categorical]).astype('float32')
    
        if not kmodel:
            k = KMeans(n_clusters=20)
            k.fit(categorical_df)
            leaf_ids = k.predict(categorical_df)
        else:
            k = kmodel
            leaf_ids = kmodel.predict(categorical_df)
            
        df2 = df[continuous_predictors].copy()
        df2['dtree_id'] = leaf_ids.astype('float32')
        df2 = df2.copy().astype('float32')
        
        # impute missing
        for c in df2.columns:
            m = df2[c].mean()
            df2[c] = df2[c].fillna(value=m)
                
        return df2, enc, k
        
    def initialize(self, toy_data, hp_types, hp_ranges):
        
        super().initialize(toy_data, hp_types, hp_ranges)
        
        for a in range(3):
            r = [random.uniform(hp_ranges[x][0], hp_ranges[x][1]) 
                 if hp_types[x] == 'float' else random.randint(hp_ranges[x][0], hp_ranges[x][1]) 
                 for x in range(len(hp_ranges))]

            rf = curfr(max_depth=self.max_depth, 
                       min_samples_leaf=self.min_samples_leaf,
                       min_samples_split=self.min_samples_split,
                       min_impurity_decrease=self.impurity_decrease,
                       n_estimators=200, accuracy_metric='rmse')
                           
            rf.fit(self.X_train, self.y_train)
            fitted_model = model.fit('split_training_data')
            self.historical_scores.append(fitted_model.score())
            self.historical_points.append(','.join([str(x) for x in r]))
            
    def fit(self, path):
                
        files = os.listdir(path)
        for f in files[:3]:
            if f[0] == 'x':
                df = self.get_file(path, f)
                df2, enc, kmodel = self.prep(df)
                
                X_train, X_test, y_train, y_test = train_test_split(df2, df['y'], test_size=0.3, random_state=42)
            
                rf = curfr(max_depth=self.max_depth, 
                           min_samples_leaf=self.min_samples_leaf,
                           min_samples_split=self.min_samples_split,
                           min_impurity_decrease=self.impurity_decrease,
                           n_estimators=200, accuracy_metric='rmse')

                rf.fit(X_train, y_train)
                
                predictions = []        
                for g in files[:3]:
                    if g[0] == 'x':
                        if g[0] != f[0]:
                            df2 = self.get_file(path, g)
                            df3 = self.prep(df2, enc, kmodel)
                            X_train2, X_test2, y_train2, y_test2 = train_test_split(df3, df2['y'], test_size=0.3, random_state=42)
                            predictions.append(rf.predict(X_test2))
                        else:
                            predictions.append(rf.predict(X_test))
                            self.truth.append(y_test.copy())

                self.all_predictions.append(predictions.copy())
                
        return self
        
    def score(self):
        sq_error = 0
        sum_of_squares = 0
        m = 0
        s = 0
        for t in self.truth:
            m += np.sum(t)
            s += len(t)
            
        for a in range(len(self.all_predictions)):
            sq_error += np.sum([(p-t)**2 for p, t in zip(self.all_predictions[a], self.truth[a])])
            sum_of_squares += np.sum([(p-m/s)**2 for p in self.all_predictions[a]])
        return 2-sq_error/sum_of_squares





#            else:
#                self.historical_scores.append(str(cross_val_score(model, toy_data.data, toy_data.target).mean()))                


#        elif self.model == 'XGBoostClassifier':
#            model = XGBClassifier(gamma=r[0], reg_lambda=r[1], colsample_bytree=r[2], 
#                                  max_depth=r[3], min_child_weight=r[4], learning_rate=r[5])
