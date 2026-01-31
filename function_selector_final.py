import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


class FunctionSelector:
    """
    Analyzes Azure Functions dataset and selects diverse functions
    for cold-start mitigation research
    """
    
    def __init__(self, data_path='./data/'):
        self.data_path = data_path
        self.invocations_df = None
        self.durations_df = None
        self.memory_df = None
        
    def load_data(self, start_day=1, end_day=3):
        """Load 3 days of data for initial analysis"""
        inv_dfs = []
        dur_dfs = []
        mem_dfs = []
        
        for day in range(start_day, end_day + 1):
            # Load invocations
            inv_file = f'{self.data_path}invocations_per_function_md.anon.d{day:02d}.csv'
            inv_df = pd.read_csv(inv_file)
            inv_df['Day'] = day
            inv_dfs.append(inv_df)
            
            # Load durations
            dur_file = f'{self.data_path}function_durations_percentiles.anon.d{day:02d}.csv'
            dur_df = pd.read_csv(dur_file)
            dur_df['Day'] = day
            dur_dfs.append(dur_df)
            
            # Load memory (if exists - only 12 days available)
            if day <= 12:
                mem_file = f'{self.data_path}app_memory_percentiles.anon.d{day:02d}.csv'
                mem_df = pd.read_csv(mem_file)
                mem_df['Day'] = day
                mem_dfs.append(mem_df)
        
        # Initialize dictionary to store data for multiple days
        self.invocations_extra_days_df = {}
        self.invocations_df = pd.concat(inv_dfs, ignore_index=True)
        self.durations_df = pd.concat(dur_dfs, ignore_index=True)
        self.memory_df = pd.concat(mem_dfs, ignore_index=True) if mem_dfs else None
        
        print(f"Loaded data: {len(self.invocations_df)} invocation records")
        print(f"Unique functions: {self.invocations_df['HashFunction'].nunique()}")
        
    def compute_function_metrics(self):
        """Compute key metrics with NaN safety and Improved Pattern Classification"""
        
        # Get minute columns (columns 1-1440)
        minute_cols = [str(i) for i in range(1, 1441)]
        
        metrics = []
        
        # Pre-calculate column indices for speed
        # Using .values is much faster than .loc for large datasets
        
        print(f"Processing {len(self.invocations_df)} functions...")
        
        for (owner, app, func, trigger), group in self.invocations_df.groupby(
            ['HashOwner', 'HashApp', 'HashFunction', 'Trigger']):
            
            # Extract invocation time series
            invocations = group[minute_cols].values.flatten()
            
            # 1. Basic Stats
            total_invocations = invocations.sum()
            mean_rate = invocations.mean()
            std_rate = invocations.std()
            max_rate = invocations.max()
            
            # 2. Safe CV Calculation (Handle mean=0)
            if mean_rate > 0:
                cv = std_rate / mean_rate
            else:
                cv = 0.0
            
            # 3. Safe PMR Calculation (Handle mean=0)
            if mean_rate > 0:
                pmr = max_rate / mean_rate
            else:
                pmr = 0.0
            
            # 4. Safe Autocorrelation (Handle flat lines / std=0)
            # This fixes the RuntimeWarning
            if std_rate > 1e-9: # Use small epsilon for float safety
                with np.errstate(all='ignore'): # Suppress warnings locally
                    ac = np.corrcoef(invocations[:-1], invocations[1:])[0, 1]
                    # Check if result is NaN (happens if one side of slice is flat)
                    if np.isnan(ac):
                        autocorr = 0.0
                    else:
                        autocorr = ac
            else:
                autocorr = 0.0
            
            # 5. Idle Percentage
            idle_minutes = (invocations == 0).sum()
            idle_pct = (idle_minutes / len(invocations)) * 100
            
            # --- IMPROVED CLASSIFICATION LOGIC ---
            
            if idle_pct > 99:
                pattern = 'sparse'
            elif cv < 0.25:
                pattern = 'steady'
            elif pmr > 5.0 or cv > 1.0:
                pattern = 'bursty'
            elif autocorr > 0.6:
                pattern = 'periodic'
            else:
                pattern = 'bursty'
            
            metrics.append({
                'HashOwner': owner,
                'HashApp': app,
                'HashFunction': func,
                'Trigger': trigger,
                'TotalInvocations': total_invocations,
                'MeanRate': mean_rate,
                'StdRate': std_rate,
                'MaxRate': max_rate,
                'CV': cv,
                'PMR': pmr,
                'Autocorr': autocorr,
                'IdlePercent': idle_pct,
                'Pattern': pattern
            })
        
        self.metrics_df = pd.DataFrame(metrics)
        
        # Merge with duration data
        if self.durations_df is not None:
            dur_agg = self.durations_df.groupby(['HashFunction']).agg({
                'Average': 'mean', 
                'Count': 'sum'
            }).reset_index()
            dur_agg.columns = ['HashFunction', 'AvgDuration', 'TotalCount']
            
            self.metrics_df = self.metrics_df.merge(dur_agg, on='HashFunction', how='left')
        
        return self.metrics_df
    
    def compute_function_memory(self):
        """
        Calculate memory per function using:
        memPerFunc = memPerApp * (durationFunc / durationApp)
        Summed over all days
        """
        print("\nComputing per-function memory estimates...")
        
        if self.memory_df is None:
            print("  ⚠ No memory data available!")
            self.metrics_df['EstimatedMemoryMB'] = np.nan
            return self.metrics_df
        
        # Group by function to get total duration per function across days
        func_durations = self.durations_df.groupby('HashFunction').agg({
            'Average': 'mean',  # Average execution time
            'Count': 'sum'      # Total executions
        }).reset_index()
        func_durations.columns = ['HashFunction', 'AvgDuration', 'TotalExecutions']
        func_durations['TotalDurationSec'] = (
            func_durations['AvgDuration'] / 1000.0  # ms to seconds
        ) * func_durations['TotalExecutions']
        
        # Group by app to get total duration per app across days
        # First, merge to get app for each function
        func_to_app = self.invocations_df[['HashFunction', 'HashApp']].drop_duplicates()
        func_durations = func_durations.merge(func_to_app, on='HashFunction', how='left')
        
        # Calculate total app duration
        app_durations = func_durations.groupby('HashApp').agg({
            'TotalDurationSec': 'sum'
        }).reset_index()
        app_durations.columns = ['HashApp', 'AppTotalDurationSec']
        
        # Get average memory per app across days
        app_memory = self.memory_df.groupby('HashApp').agg({
            'AverageAllocatedMb': 'mean'
        }).reset_index()
        app_memory.columns = ['HashApp', 'AvgAppMemoryMB']
        
        # Merge everything together
        func_durations = func_durations.merge(app_durations, on='HashApp', how='left')
        func_durations = func_durations.merge(app_memory, on='HashApp', how='left')
        
        # Calculate per-function memory estimate
        # memPerFunc = memPerApp * (durationFunc / durationApp)
        func_durations['EstimatedMemoryMB'] = (
            func_durations['AvgAppMemoryMB'] * 
            (func_durations['TotalDurationSec'] / func_durations['AppTotalDurationSec'])
        )
        
        # Handle edge cases
        func_durations['EstimatedMemoryMB'] = func_durations['EstimatedMemoryMB'].fillna(0)
        
        # Merge back to metrics
        memory_cols = func_durations[['HashFunction', 'EstimatedMemoryMB']]
        self.metrics_df = self.metrics_df.merge(memory_cols, on='HashFunction', how='left')
        self.metrics_df['EstimatedMemoryMB'] = self.metrics_df['EstimatedMemoryMB'].fillna(0)
        
        print(f"  ✓ Computed memory for {(self.metrics_df['EstimatedMemoryMB'] > 0).sum()} functions")
        print(f"  Memory range: {self.metrics_df['EstimatedMemoryMB'].min():.2f} - "
            f"{self.metrics_df['EstimatedMemoryMB'].max():.2f} MB")
        
        return self.metrics_df

    def categorize_functions(self):
        """Categorize functions by frequency and duration using Q1/Q3 (High/Low only)."""
        # Frequency: Q1 and Q3 of TotalInvocations
        freq_q1 = self.metrics_df['TotalInvocations'].quantile(0.25)
        freq_q3 = self.metrics_df['TotalInvocations'].quantile(0.75)

        def freq_label(x):
            if x <= freq_q1:
                return 'Low'    # bottom 25% = low frequency
            elif x >= freq_q3:
                return 'High'   # top 25% = high frequency
            else:
                return 'medium'     # middle 50%, can be ignored later

        self.metrics_df['FrequencyCategory'] = self.metrics_df['TotalInvocations'].apply(freq_label)

        # Duration: Q1 and Q3 of AvgDuration (High = Slow, Low = Fast)
        dur_q1 = self.metrics_df['AvgDuration'].quantile(0.25)
        dur_q3 = self.metrics_df['AvgDuration'].quantile(0.75)

        def dur_label(x):
            if x <= dur_q1:
                return 'Low'    # Low = Fast (bottom 25% of duration)
            elif x >= dur_q3:
                return 'High'   # High = Slow (top 25% of duration)
            else:
                return 'medium'     # middle 50%, can be ignored later

        self.metrics_df['DurationCategory'] = self.metrics_df['AvgDuration'].apply(dur_label)

    def categorize_memory(self):
        """Categorize functions by memory usage using Q1 and Q3 (Low/Med/High)."""
        if 'EstimatedMemoryMB' not in self.metrics_df.columns:
            print("  ⚠ No memory data to categorize")
            return
        
        # Remove zeros for categorization
        non_zero_memory = self.metrics_df[self.metrics_df['EstimatedMemoryMB'] > 0]['EstimatedMemoryMB']
        
        if len(non_zero_memory) > 0:
            # Compute Q1 (25th percentile) and Q3 (75th percentile)
            q1 = non_zero_memory.quantile(0.25)
            q3 = non_zero_memory.quantile(0.75)
            
            # Optional: enforce minimum spacing so Q1 != Q3
            if q1 == q3:
                # fallback to a simple below/above threshold using that value
                self.metrics_df['MemoryCategory'] = pd.cut(
                    self.metrics_df['EstimatedMemoryMB'],
                    bins=[0, q1, float('inf')],
                    labels=['Low', 'High'],
                    include_lowest=True
                )
                print(f"  Memory threshold (fallback): Low < {q1:.1f} MB, High ≥ {q1:.1f} MB")
                return
            
            # 3-level categorization: Low, Medium, High
            self.metrics_df['MemoryCategory'] = pd.cut(
                self.metrics_df['EstimatedMemoryMB'],
                bins=[0, q1, q3, float('inf')],
                labels=['Low', 'Medium', 'High'],
                include_lowest=True
            )
            
            print(
                f"  Memory thresholds: "
                f"Low < {q1:.1f} MB, "
                f"Medium [{q1:.1f}, {q3:.1f}) MB, "
                f"High ≥ {q3:.1f} MB"
            )
        else:
            self.metrics_df['MemoryCategory'] = 'Unknown'

    # select 1 steady function 
    def select_steady_function(self, http_only=True):
        """
        Select a steady function (low CV) for cold-start mitigation research.
        """
        print("\nSelecting steady function...")
        
        if http_only:
            candidates = self.metrics_df[
                (self.metrics_df['Trigger'] == 'http') &
                (self.metrics_df['Pattern'] == 'steady') &
                (self.metrics_df['TotalInvocations'] > 50)
            ].copy()
            print(f"  Filtering: {len(candidates)} HTTP steady functions with >50 invocations")
        else:
            candidates = self.metrics_df[
                (self.metrics_df['Pattern'] == 'steady') &
                (self.metrics_df['TotalInvocations'] > 50)
            ].copy()
            print(f"  Filtering: {len(candidates)} steady functions with >50 invocations")
        
        if len(candidates) == 0:
            print("  ⚠ No steady functions found!")
            return pd.DataFrame()
        
        # Select the function with the lowest CV
        selected_function = candidates.sort_values('CV', ascending=True).head(1)
        
        print(f"\n{'='*60}")
        print(f"  ✓ Selected steady function:")
        print(selected_function[['HashFunction', 'Trigger', 'Pattern', 'TotalInvocations', 'CV']])
        print(f"{'='*60}")
        
        return selected_function


    def select_diverse_functions_rand(self, n_functions=10, http_only=True, include_memory=True, random_seed=None):
        """
        Select functions for cold-start mitigation research.
        
        Strategy (for n=10):
        - 8 BURSTY functions: Memory (H/L) × Duration (H/L) × Invocations (H/L) = 8 combinations
        - 1 PERIODIC function: baseline for predictable workloads
        - 1 SPARSE function: challenge case (always cold)
        
        Parameters:
        -----------
        random_seed : int, optional
            If provided, randomly select from each cluster instead of picking top by invocations.
            Use different seeds to generate different function sets.
        """
        
        # Set random seed if provided
        if random_seed is not None:
            np.random.seed(random_seed)
            print(f"  🎲 Random seed: {random_seed} (for reproducibility)")
        
        selected = []
        
        # Helper to safely append
        def safe_append(df, k=1, label="", randomize=False):
            nonlocal selected
            if len(df) > 0:
                # Filter out zero memory functions
                if 'EstimatedMemoryMB' in df.columns:
                    df = df[df['EstimatedMemoryMB'] > 0]
                
                if len(df) == 0:
                    print(f"    ✗ No valid match for: {label}")
                    return
                
                # Select function(s)
                if randomize and random_seed is not None:
                    # Random selection from cluster
                    sample_size = min(k, len(df))
                    sampled = df.sample(n=sample_size, random_state=random_seed)
                    for _, row in sampled.iterrows():
                        if len(selected) < n_functions:
                            selected.append(row)
                            if label:
                                print(f"    ✓ Added (random): {label}")
                else:
                    # Deterministic selection (top k)
                    for _, row in df.head(k).iterrows():
                        if len(selected) < n_functions:
                            selected.append(row)
                            if label:
                                print(f"    ✓ Added: {label}")
            else:
                print(f"    ✗ No match for: {label}")
        
        # ============================
        # Phase A: Filter HTTP only + exclude mediums
        # ============================
        
        if http_only:
            base_df = self.metrics_df[
                (self.metrics_df['Trigger'] == 'http') &
                (self.metrics_df['TotalInvocations'] > 50) &
                (self.metrics_df['FrequencyCategory'] != 'medium') &
                (self.metrics_df['MemoryCategory'] != 'medium') &
                (self.metrics_df['DurationCategory'] != 'medium')
            ].copy()
            print(f"  Filtering: {len(base_df)} HTTP functions with >50 invocations (excluding medium frequency)")
        else:
            base_df = self.metrics_df[
                (self.metrics_df['TotalInvocations'] > 50) &
                (self.metrics_df['FrequencyCategory'] != 'medium')&
                (self.metrics_df['MemoryCategory'] != 'medium') &
                (self.metrics_df['DurationCategory'] != 'medium')
            ].copy()
        
        if len(base_df) == 0:
            print("  ⚠ No functions meet criteria!")
            return pd.DataFrame()
        
        # ============================
        # Phase B: 8 BURSTY functions (2×2×2)
        # ============================
        
        bursty = base_df[base_df['Pattern'] == 'bursty'].copy()
        print(f"\n  Found {len(bursty)} bursty functions")
        
        if include_memory and 'MemoryCategory' in bursty.columns and 'DurationCategory' in bursty.columns:
            print("  Selecting 8 bursty functions (2 memory × 2 duration × 2 invocations)...")
            
            # All 8 combinations
            combinations = [
                ('High', 'High', 'High', 'High Mem + High Dur + High Inv (worst case)'),
                ('High', 'High', 'Low', 'High Mem + High Dur + Low Inv'),
                ('High', 'Low', 'High', 'High Mem + Low Dur + High Inv'),
                ('High', 'Low', 'Low', 'High Mem + Low Dur + Low Inv'),
                ('Low', 'High', 'High', 'Low Mem + High Dur + High Inv'),
                ('Low', 'High', 'Low', 'Low Mem + High Dur + Low Inv'),
                ('Low', 'Low', 'High', 'Low Mem + Low Dur + High Inv (best case)'),
                ('Low', 'Low', 'Low', 'Low Mem + Low Dur + Low Inv'),
            ]
            
            for mem, dur, inv, label in combinations:
                # Filter candidates
                candidates = bursty[
                    (bursty['MemoryCategory'] == mem) &
                    (bursty['DurationCategory'] == dur) &
                    (bursty['FrequencyCategory'] == inv)
                ].sort_values('TotalInvocations', ascending=False)
                
                safe_append(candidates, k=1, label=label, randomize=(random_seed is not None))
        else:
            print("  ⚠ Missing memory/duration/frequency categories, selecting top bursty by metrics")
            safe_append(bursty.sort_values('CV', ascending=False), k=8, label="Top bursty", randomize=(random_seed is not None))
        
        # ============================
        # Phase C: 1 PERIODIC function
        # ============================
        
        print(f"\n  Selecting 1 periodic function (predictable baseline)...")
        
        periodic = base_df[base_df['Pattern'] == 'periodic'].copy()
        print(f"  Found {len(periodic)} periodic functions")
        
        if len(periodic) > 0:
            safe_append(
                periodic.sort_values('Autocorr', ascending=False),
                k=1, label="Periodic (high autocorr, ARIMA-friendly)",
                randomize=(random_seed is not None)
            )
        else:
            print("  ⚠ No periodic functions found, filling with bursty")
            selected_hashes = {row['HashFunction'] for row in selected}
            remaining_bursty = bursty[~bursty['HashFunction'].isin(selected_hashes)]
            safe_append(remaining_bursty.sort_values('CV', ascending=False), k=1, label="Extra bursty (fallback)", randomize=(random_seed is not None))
        
        # ============================
        # Phase D: 1 SPARSE function
        # ============================
        
        print(f"\n  Selecting 1 sparse function (always cold, worst case)...")
        
        sparse = base_df[base_df['Pattern'] == 'sparse'].copy()
        print(f"  Found {len(sparse)} sparse functions")
        
        if len(sparse) > 0:
            safe_append(
                sparse.sort_values('IdlePercent', ascending=False),
                k=10, label="Sparse (highest idle %, always cold)",
                randomize=(random_seed is not None)
            )
        else:
            print("  ⚠ No sparse functions found, filling with bursty")
            selected_hashes = {row['HashFunction'] for row in selected}
            remaining_bursty = bursty[~bursty['HashFunction'].isin(selected_hashes)]
            safe_append(remaining_bursty.sort_values('IdlePercent', ascending=False), k=1, label="Extra bursty (fallback)", randomize=(random_seed is not None))

        # Phase D.A 1 STEADY function
        print(f"\n  Selecting 1 steady function (consistent workload)...")
        steady = base_df[base_df['Pattern'] == 'steady'].copy()
        print(f"  Found {len(steady)} steady functions")
        if len(steady) > 0:
            safe_append(
                steady.sort_values('CV', ascending=True),
                k=1, label="Steady (low CV, consistent workload)",
                randomize=(random_seed is not None)
            )
        else:
            print("  ⚠ No steady functions found, filling with bursty")
            selected_hashes = {row['HashFunction'] for row in selected}
            remaining_bursty = bursty[~bursty['HashFunction'].isin(selected_hashes)]
            safe_append(remaining_bursty.sort_values('CV', ascending=True), k=1, label="Extra bursty (fallback)", randomize=(random_seed is not None))

        
        # ============================
        # Phase E: Fill if still short
        # ============================
        
        if len(selected) < n_functions:
            print(f"\n  ⚠ Only {len(selected)}/{n_functions} functions selected, filling gaps...")
            selected_hashes = {row['HashFunction'] for row in selected}
            
            remaining_bursty = bursty[~bursty['HashFunction'].isin(selected_hashes)]
            safe_append(
                remaining_bursty.sort_values('TotalInvocations', ascending=False),
                k=n_functions - len(selected),
                label="Additional bursty (gap fill)",
                randomize=(random_seed is not None)
            )
            
            if len(selected) < n_functions:
                selected_hashes = {row['HashFunction'] for row in selected}
                remaining = base_df[~base_df['HashFunction'].isin(selected_hashes)]
                safe_append(
                    remaining.sort_values('TotalInvocations', ascending=False),
                    k=n_functions - len(selected),
                    label="Any HTTP (last resort)",
                    randomize=(random_seed is not None)
                )
        
        # ============================
        # Finalize
        # ============================
        
        self.selected_functions = pd.DataFrame(selected).head(n_functions)
        
        print(f"\n{'='*60}")
        print(f"  ✓ Selected {len(self.selected_functions)} functions")
        print(f"{'='*60}")
        
        if len(self.selected_functions) > 0:
            print(f"  Pattern distribution: {self.selected_functions['Pattern'].value_counts().to_dict()}")
            if 'MemoryCategory' in self.selected_functions.columns:
                print(f"  Memory distribution: {self.selected_functions['MemoryCategory'].value_counts().to_dict()}")
            if 'DurationCategory' in self.selected_functions.columns:
                print(f"  Duration distribution: {self.selected_functions['DurationCategory'].value_counts().to_dict()}")
            if 'FrequencyCategory' in self.selected_functions.columns:
                print(f"  Frequency distribution: {self.selected_functions['FrequencyCategory'].value_counts().to_dict()}")
        
        return self.selected_functions
    
    def compute_percentile_data(self):
        """
        Compute and store duration and memory percentiles for selected functions
        Returns DataFrames with percentile data for each function
        """
        print("\nComputing percentile data for selected functions...")
        
        if self.selected_functions is None or len(self.selected_functions) == 0:
            print("  ⚠ No functions selected yet!")
            return None, None
        
        # Get list of selected function hashes
        selected_hashes = self.selected_functions['HashFunction'].tolist()
        
        # ============================
        # Duration Percentiles
        # ============================
        
        duration_percentiles = self.durations_df[
            self.durations_df['HashFunction'].isin(selected_hashes)
        ].copy()
        
        print(f"  ✓ Duration percentiles: {len(duration_percentiles)} records")
        
        # # ============================
        # # Memory Percentiles (App-level)
        # # ============================
        
        # memory_percentiles = None
        # if self.memory_df is not None:
        #     # Get the apps for selected functions
        #     selected_apps = self.selected_functions['HashApp'].unique().tolist()
            
        #     memory_percentiles = self.memory_df[
        #         self.memory_df['HashApp'].isin(selected_apps)
        #     ].copy()
            
        #     # Merge to add function hash for reference
        #     func_to_app = self.invocations_df[['HashFunction', 'HashApp']].drop_duplicates()
        #     memory_percentiles = memory_percentiles.merge(
        #         func_to_app, 
        #         on='HashApp', 
        #         how='left'
        #     )
            
        #     # Filter to only selected functions
        #     memory_percentiles = memory_percentiles[
        #         memory_percentiles['HashFunction'].isin(selected_hashes)
        #     ]
            
        #     print(f"  ✓ Memory percentiles: {len(memory_percentiles)} records")
        # else:
        #     print("  ⚠ No memory percentile data available")

            # ============================
        # Memory Percentiles (Function-level)
        # ============================
        
        memory_percentiles = None
        if self.memory_df is not None:
            print("  Computing per-function memory percentiles...")
            
            # Get mapping of function to app
            func_to_app = self.invocations_df[['HashFunction', 'HashApp']].drop_duplicates()
            
            # Get duration data for ALL functions (need app totals)
            func_durations = self.durations_df.groupby('HashFunction').agg({
                'Average': 'mean',
                'Count': 'sum'
            }).reset_index()
            func_durations.columns = ['HashFunction', 'AvgDuration', 'TotalExecutions']
            func_durations['TotalDurationSec'] = (
                func_durations['AvgDuration'] / 1000.0
            ) * func_durations['TotalExecutions']
            
            # Merge with app info
            func_durations = func_durations.merge(func_to_app, on='HashFunction', how='left')
            
            # Calculate total app duration
            app_durations = func_durations.groupby('HashApp').agg({
                'TotalDurationSec': 'sum'
            }).reset_index()
            app_durations.columns = ['HashApp', 'AppTotalDurationSec']
            
            # Merge back
            func_durations = func_durations.merge(app_durations, on='HashApp', how='left')
            func_durations['DurationRatio'] = (
                func_durations['TotalDurationSec'] / func_durations['AppTotalDurationSec']
            )
            
            # Now calculate memory percentiles for each selected function
            memory_percentile_records = []
            
            for func_hash in selected_hashes:
                # Get the app and duration ratio for this function
                func_info = func_durations[func_durations['HashFunction'] == func_hash]
                
                if len(func_info) == 0:
                    print(f"    ⚠ No duration data for function {func_hash[:16]}...")
                    continue
                
                func_app = func_info.iloc[0]['HashApp']
                duration_ratio = func_info.iloc[0]['DurationRatio']
                
                # Get app memory percentiles
                app_mem = self.memory_df[self.memory_df['HashApp'] == func_app]
                
                if len(app_mem) == 0:
                    print(f"    ⚠ No memory data for app {func_app[:16]}...")
                    continue
                
                # For each day, calculate function memory percentiles
                for _, app_row in app_mem.iterrows():
                    func_mem_record = {
                        'HashFunction': func_hash,
                        'HashApp': func_app,
                        'Day': app_row['Day']
                    }
                    
                    # Calculate memory percentiles: memPerFunc = memPerApp * durationRatio
                    # Apply to all percentile columns
                    percentile_cols = [col for col in app_mem.columns 
                                    if col.startswith('Average') or 
                                        col.startswith('Count') or 
                                        col.startswith('Percentile') or
                                        col.endswith('Mb')]
                    
                    for col in percentile_cols:
                        if col in app_row.index and pd.notna(app_row[col]):
                            func_mem_record[col] = app_row[col] * duration_ratio
                        else:
                            func_mem_record[col] = 0
                    
                    memory_percentile_records.append(func_mem_record)
            
            if len(memory_percentile_records) > 0:
                memory_percentiles = pd.DataFrame(memory_percentile_records)
                print(f"  ✓ Memory percentiles: {len(memory_percentiles)} records for {len(selected_hashes)} functions")
            else:
                print("  ⚠ No memory percentiles computed")
        else:
            print("  ⚠ No memory data available")
        
        self.duration_percentiles = duration_percentiles
        self.memory_percentiles = memory_percentiles
        
        return duration_percentiles, memory_percentiles

    def visualize_selected_functions(self):
        """Create visualization dashboard for selected functions"""
        
        has_memory = 'EstimatedMemoryMB' in self.selected_functions.columns
        
        if has_memory:
            fig, axes = plt.subplots(4, 3, figsize=(18, 16))
        else:
            fig, axes = plt.subplots(3, 3, figsize=(18, 12))
        
        fig.suptitle('Selected Functions Overview', fontsize=16, y=0.995)
        
        # 1. Trigger distribution
        axes[0, 0].bar(
            self.selected_functions['Trigger'].value_counts().index,
            self.selected_functions['Trigger'].value_counts().values
        )
        axes[0, 0].set_title('Trigger Types')
        axes[0, 0].set_xlabel('Trigger')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Frequency distribution
        # 2. Frequency distribution
        freq_order = ['Low', 'High']
        freq_counts = self.selected_functions['FrequencyCategory'].value_counts()
        axes[0, 1].bar(freq_order, [freq_counts.get(f, 0) for f in freq_order])
        axes[0, 1].set_title('Frequency Categories')
        axes[0, 1].set_xlabel('Frequency')
        axes[0, 1].set_ylabel('Count')
        
        # 3. Pattern distribution
        pattern_counts = self.selected_functions['Pattern'].value_counts()
        axes[0, 2].bar(pattern_counts.index, pattern_counts.values)
        axes[0, 2].set_title('Pattern Types')
        axes[0, 2].set_xlabel('Pattern')
        axes[0, 2].set_ylabel('Count')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # 4. Total invocations
        axes[1, 0].barh(range(len(self.selected_functions)), 
                        self.selected_functions['TotalInvocations'])
        axes[1, 0].set_yticks(range(len(self.selected_functions)))
        axes[1, 0].set_yticklabels([f"F{i+1}" for i in range(len(self.selected_functions))])
        axes[1, 0].set_title('Total Invocations (3 days)')
        axes[1, 0].set_xlabel('Invocations')
        
        # 5. Average duration
        axes[1, 1].barh(range(len(self.selected_functions)), 
                        self.selected_functions['AvgDuration'])
        axes[1, 1].set_yticks(range(len(self.selected_functions)))
        axes[1, 1].set_yticklabels([f"F{i+1}" for i in range(len(self.selected_functions))])
        axes[1, 1].set_title('Average Execution Time (ms)')
        axes[1, 1].set_xlabel('Duration (ms)')
        
        # 6. Memory (if available)
        if has_memory:
            axes[1, 2].barh(range(len(self.selected_functions)), 
                            self.selected_functions['EstimatedMemoryMB'])
            axes[1, 2].set_yticks(range(len(self.selected_functions)))
            axes[1, 2].set_yticklabels([f"F{i+1}" for i in range(len(self.selected_functions))])
            axes[1, 2].set_title('Estimated Memory (MB)')
            axes[1, 2].set_xlabel('Memory (MB)')
        else:
            # Coefficient of Variation
            axes[1, 2].barh(range(len(self.selected_functions)), 
                            self.selected_functions['CV'])
            axes[1, 2].set_yticks(range(len(self.selected_functions)))
            axes[1, 2].set_yticklabels([f"F{i+1}" for i in range(len(self.selected_functions))])
            axes[1, 2].set_title('Coefficient of Variation')
            axes[1, 2].set_xlabel('CV')
        
        # 7. Idle percentage
        axes[2, 0].barh(range(len(self.selected_functions)), 
                        self.selected_functions['IdlePercent'])
        axes[2, 0].set_yticks(range(len(self.selected_functions)))
        axes[2, 0].set_yticklabels([f"F{i+1}" for i in range(len(self.selected_functions))])
        axes[2, 0].set_title('Idle Time (%)')
        axes[2, 0].set_xlabel('Idle %')
        
        # 8. Autocorrelation
        axes[2, 1].barh(range(len(self.selected_functions)), 
                        self.selected_functions['Autocorr'])
        axes[2, 1].set_yticks(range(len(self.selected_functions)))
        axes[2, 1].set_yticklabels([f"F{i+1}" for i in range(len(self.selected_functions))])
        axes[2, 1].set_title('Autocorrelation (Periodicity)')
        axes[2, 1].set_xlabel('Autocorr')
        
        # 9. Summary table
        axes[2, 2].axis('tight')
        axes[2, 2].axis('off')
        summary_data = []
        for i, row in self.selected_functions.iterrows():
            summary_data.append([
                f"F{len(summary_data)+1}",
                row['Trigger'][:4],
                row['Pattern'][:4],
                f"{row['TotalInvocations']:.0f}"
            ])
        
        table = axes[2, 2].table(cellText=summary_data,
                                colLabels=['ID', 'Trig', 'Ptrn', 'Inv'],
                                cellLoc='center',
                                loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Additional memory visualizations
        if has_memory:
            # 10. Memory category distribution
            if 'MemoryCategory' in self.selected_functions.columns:
                mem_order = ['Low', 'Medium', 'High']
                mem_counts = self.selected_functions['MemoryCategory'].value_counts()
                axes[3, 0].bar(mem_order, [mem_counts.get(m, 0) for m in mem_order])
                axes[3, 0].set_title('Memory Categories')
                axes[3, 0].set_xlabel('Memory')
                axes[3, 0].set_ylabel('Count')
            
            # 11. Memory vs Invocations scatter
            axes[3, 1].scatter(
                self.selected_functions['TotalInvocations'],
                self.selected_functions['EstimatedMemoryMB'],
                alpha=0.6, s=100
            )
            axes[3, 1].set_xlabel('Total Invocations')
            axes[3, 1].set_ylabel('Memory (MB)')
            axes[3, 1].set_title('Memory vs Frequency')
            axes[3, 1].grid(True, alpha=0.3)
            
            # 12. Memory vs Duration scatter
            axes[3, 2].scatter(
                self.selected_functions['AvgDuration'],
                self.selected_functions['EstimatedMemoryMB'],
                alpha=0.6, s=100, c=range(len(self.selected_functions)), cmap='viridis'
            )
            axes[3, 2].set_xlabel('Avg Duration (ms)')
            axes[3, 2].set_ylabel('Memory (MB)')
            axes[3, 2].set_title('Memory vs Duration')
            axes[3, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

    def export_selected_data(self, output_path='./selected_functions/', extra_days_range=[10,12]):
        """
        Export time series data, metadata, and percentiles for selected functions
        
        Parameters:
        -----------
        output_path : str, default './selected_functions/'
            Path to export data
        extra_days_range : tuple or None, optional
            Range of days to export as (start_day, end_day), e.g., (9, 12)
            If None, no extra days are exported
        """
        import os
        os.makedirs(output_path, exist_ok=True)
        
        # ============================
        # 1. Export metadata
        # ============================
        self.selected_functions.to_csv(
            f'{output_path}selected_functions_metadata.csv', 
            index=False
        )
        print(f"  ✓ Exported metadata")
        
        # ============================
        # 2. Export invocation time series for each function
        # ============================
        minute_cols = [str(i) for i in range(1, 1441)]
        
        for idx, row in self.selected_functions.iterrows():
            func_hash = row['HashFunction']
            
            # Get invocation time series
            func_inv = self.invocations_df[
                self.invocations_df['HashFunction'] == func_hash
            ][['Day'] + minute_cols]
            
            func_inv.to_csv(
                f'{output_path}function_{idx+1}_invocations.csv',
                index=False
            )
        
        print(f"  ✓ Exported invocation time series for {len(self.selected_functions)} functions")
        
        # ============================
        # 2B. Export extra days data in a single CSV file (if range provided)
        # ============================
        if extra_days_range is not None:
            start_day, end_day = extra_days_range
            print(f"\n  Loading data for days {start_day}-{end_day}...")
            
            # Load data for the specified range
            extra_days_dfs = {}
            for day in range(start_day, end_day + 1):
                try:
                    day_df = pd.read_csv(f'{self.data_path}invocations_per_function_md.anon.d{day:02d}.csv')
                    day_df['Day'] = day
                    extra_days_dfs[day] = day_df
                    print(f"  ✓ Loaded day {day}")
                except FileNotFoundError:
                    print(f"  ⚠ Day {day} data not found")
            
            # Combine all days and export as single CSV per function
            for idx, row in self.selected_functions.iterrows():
                func_hash = row['HashFunction']
                
                # Collect data from all days for this function
                func_inv_all_days = []
                for day, day_df in extra_days_dfs.items():
                    func_inv_day = day_df[
                        day_df['HashFunction'] == func_hash
                    ][['Day'] + minute_cols]
                    
                    if len(func_inv_day) > 0:
                        func_inv_all_days.append(func_inv_day)
                
                # Combine all days into single DataFrame
                if func_inv_all_days:
                    func_inv_combined = pd.concat(func_inv_all_days, ignore_index=True)
                    func_inv_combined.to_csv(
                        f'{output_path}function_{idx+1}_invocations_days_{start_day:02d}_{end_day:02d}.csv',
                        index=False
                    )
            
            print(f"  ✓ Exported invocation time series for days {start_day}-{end_day} for {len(self.selected_functions)} functions")
        
        # ============================
        # 3. Export duration percentiles for each function
        # ============================
        if hasattr(self, 'duration_percentiles') and self.duration_percentiles is not None:
            for idx, row in self.selected_functions.iterrows():
                func_hash = row['HashFunction']
                
                # Get duration percentiles for this function
                func_dur = self.duration_percentiles[
                    self.duration_percentiles['HashFunction'] == func_hash
                ]
                
                if len(func_dur) > 0:
                    func_dur.to_csv(
                        f'{output_path}function_{idx+1}_duration_percentiles.csv',
                        index=False
                    )
            
            print(f"  ✓ Exported duration percentiles for {len(self.selected_functions)} functions")
        else:
            print("  ⚠ No duration percentiles to export (run compute_percentile_data() first)")
        
        # ============================
        # 4. Export memory percentiles for each function
        # ============================
        if hasattr(self, 'memory_percentiles') and self.memory_percentiles is not None:
            for idx, row in self.selected_functions.iterrows():
                func_hash = row['HashFunction']
                
                # Get memory percentiles for this function's app
                func_mem = self.memory_percentiles[
                    self.memory_percentiles['HashFunction'] == func_hash
                ]
                
                if len(func_mem) > 0:
                    func_mem.to_csv(
                        f'{output_path}function_{idx+1}_memory_percentiles.csv',
                        index=False
                    )
            
            print(f"  ✓ Exported memory percentiles for {len(self.selected_functions)} functions")
        else:
            print("  ⚠ No memory percentiles to export")
        
        print(f"\n{'='*60}")
        print(f"Exported all data to {output_path}")
        print(f"{'='*60}")
        print(f"Files per function:")
        print(f"  - function_N_invocations.csv (time series)")
        print(f"  - function_N_duration_percentiles.csv (all percentiles)")
        print(f"  - function_N_memory_percentiles.csv (all percentiles)")
        print(f"Plus:")
        print(f"  - selected_functions_metadata.csv (summary)")

    def print_selection_matrix(self):
        """
        Print selection breakdown showing 8 bursty combinations + 1 periodic + 1 sparse
        """
        if self.selected_functions is None or len(self.selected_functions) == 0:
            print("No functions selected yet!")
            return
        
        print("\n" + "="*80)
        print("FUNCTION SELECTION BREAKDOWN (8 Bursty + 1 Periodic + 1 Sparse)")
        print("="*80)
        
        # Show bursty functions in a structured way
        bursty_funcs = self.selected_functions[self.selected_functions['Pattern'] == 'bursty']
        
        if len(bursty_funcs) > 0 and 'MemoryCategory' in bursty_funcs.columns:
            print("\nBURSTY FUNCTIONS (8 combinations):")
            print("-" * 80)
            
            for idx, row in bursty_funcs.iterrows():
                func_id = idx + 1
                mem = row.get('MemoryCategory', '?')
                dur = row.get('DurationCategory', '?')
                inv = row.get('FrequencyCategory', '?')
                print(f"  F{func_id}: Memory={mem:4s} | Duration={dur:4s} | Invocations={inv:4s}")
        
        # Show periodic
        periodic_funcs = self.selected_functions[self.selected_functions['Pattern'] == 'periodic']
        if len(periodic_funcs) > 0:
            print("\nPERIODIC FUNCTION (baseline):")
            print("-" * 80)
            for idx, row in periodic_funcs.iterrows():
                func_id = idx + 1
                print(f"  F{func_id}: Autocorr={row['Autocorr']:.3f}")
        
        # Show sparse
        sparse_funcs = self.selected_functions[self.selected_functions['Pattern'] == 'sparse']
        if len(sparse_funcs) > 0:
            print("\nSPARSE FUNCTION (always cold):")
            print("-" * 80)
            for idx, row in sparse_funcs.iterrows():
                func_id = idx + 1
                print(f"  F{func_id}: Idle={row['IdlePercent']:.1f}%")
        
        print("\n" + "="*80)

    def plot_http_distributions(self, output_path='./'):
        """
        Plot and save histograms showing distributions of HTTP functions
        with threshold lines for High/Low categorization
        """
        import os
        
        print("\nGenerating HTTP function distribution plots...")
        
        # Filter to HTTP functions only
        http_funcs = self.metrics_df[
            (self.metrics_df['Trigger'] == 'http') &
            (self.metrics_df['TotalInvocations'] > 50)
        ].copy()
        
        if len(http_funcs) == 0:
            print("  ⚠ No HTTP functions found!")
            return
        
        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('HTTP Functions Distribution with High/Low Thresholds', 
                    fontsize=16, fontweight='bold')
        
        # ============================
        # 1. Total Invocations
        # ============================
        ax = axes[0]
        
        # Calculate threshold
        freq_threshold = http_funcs['TotalInvocations'].quantile(0.75)
        
        # Plot histogram
        ax.hist(http_funcs['TotalInvocations'], bins=50, alpha=0.7, 
                color='steelblue', edgecolor='black')
        
        # Add threshold line
        ax.axvline(freq_threshold, color='red', linestyle='--', linewidth=2, 
                label=f'High/Low threshold: {freq_threshold:.0f}')
        
        # Labels
        ax.set_xlabel('Total Invocations (3 days)', fontsize=12)
        ax.set_ylabel('Number of Functions', fontsize=12)
        ax.set_title('Invocations Distribution', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add counts
        low_count = (http_funcs['TotalInvocations'] <= freq_threshold).sum()
        high_count = (http_funcs['TotalInvocations'] > freq_threshold).sum()
        ax.text(0.02, 0.98, f'Low: {low_count}\nHigh: {high_count}', 
                transform=ax.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # ============================
        # 2. Duration
        # ============================
        ax = axes[1]
        
        # Calculate threshold
        dur_threshold = http_funcs['AvgDuration'].median()
        
        # Plot histogram
        ax.hist(http_funcs['AvgDuration'], bins=50, alpha=0.7, 
                color='orange', edgecolor='black')
        
        # Add threshold line
        ax.axvline(dur_threshold, color='red', linestyle='--', linewidth=2,
                label=f'High/Low threshold: {dur_threshold:.1f} ms')
        
        # Labels
        ax.set_xlabel('Average Duration (ms)', fontsize=12)
        ax.set_ylabel('Number of Functions', fontsize=12)
        ax.set_title('Duration Distribution', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add counts
        low_count = (http_funcs['AvgDuration'] <= dur_threshold).sum()
        high_count = (http_funcs['AvgDuration'] > dur_threshold).sum()
        ax.text(0.02, 0.98, f'Low: {low_count}\nHigh: {high_count}', 
                transform=ax.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # ============================
        # 3. Memory
        # ============================
        ax = axes[2]
        
        if 'EstimatedMemoryMB' in http_funcs.columns:
            # Filter out zero memory
            non_zero_mem = http_funcs[http_funcs['EstimatedMemoryMB'] > 0]['EstimatedMemoryMB']
            
            if len(non_zero_mem) > 0:
                # Calculate threshold
                mem_threshold = non_zero_mem.median()
                
                # Plot histogram
                ax.hist(non_zero_mem, bins=50, alpha=0.7, 
                        color='green', edgecolor='black')
                
                # Add threshold line
                ax.axvline(mem_threshold, color='red', linestyle='--', linewidth=2,
                        label=f'High/Low threshold: {mem_threshold:.1f} MB')
                
                # Labels
                ax.set_xlabel('Estimated Memory (MB)', fontsize=12)
                ax.set_ylabel('Number of Functions', fontsize=12)
                ax.set_title('Memory Distribution', fontsize=14, fontweight='bold')
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.3)
                
                # Add counts
                low_count = (non_zero_mem <= mem_threshold).sum()
                high_count = (non_zero_mem > mem_threshold).sum()
                zero_count = (http_funcs['EstimatedMemoryMB'] == 0).sum()
                ax.text(0.02, 0.98, f'Low: {low_count}\nHigh: {high_count}\nZero: {zero_count}', 
                        transform=ax.transAxes, fontsize=11, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            else:
                ax.text(0.5, 0.5, 'No memory data available', 
                        transform=ax.transAxes, ha='center', va='center', fontsize=14)
                ax.set_title('Memory Distribution', fontsize=14, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No memory data available', 
                    transform=ax.transAxes, ha='center', va='center', fontsize=14)
            ax.set_title('Memory Distribution', fontsize=14, fontweight='bold')
        
        # ============================
        # Save figure
        # ============================
        plt.tight_layout()
        
        output_file = os.path.join(output_path, 'http_functions_distributions.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        
        print(f"  ✓ Saved distribution plot to: {output_file}")
        
        # Print summary statistics
        print(f"\n{'='*60}")
        print("HTTP FUNCTIONS SUMMARY STATISTICS")
        print(f"{'='*60}")
        print(f"Total HTTP functions (>50 invocations): {len(http_funcs)}")
        print(f"\nInvocations:")
        print(f"  Min: {http_funcs['TotalInvocations'].min():.0f}")
        print(f"  Median: {http_funcs['TotalInvocations'].median():.0f}")
        print(f"  75th percentile (threshold): {freq_threshold:.0f}")
        print(f"  Max: {http_funcs['TotalInvocations'].max():.0f}")
        print(f"\nDuration (ms):")
        print(f"  Min: {http_funcs['AvgDuration'].min():.1f}")
        print(f"  Median (threshold): {dur_threshold:.1f}")
        print(f"  Max: {http_funcs['AvgDuration'].max():.1f}")
        
        if 'EstimatedMemoryMB' in http_funcs.columns and len(non_zero_mem) > 0:
            print(f"\nMemory (MB):")
            print(f"  Min: {non_zero_mem.min():.2f}")
            print(f"  Median (threshold): {mem_threshold:.2f}")
            print(f"  Max: {non_zero_mem.max():.2f}")
            print(f"  Functions with zero memory: {(http_funcs['EstimatedMemoryMB'] == 0).sum()}")
        print(f"{'='*60}\n")
        
        return fig

    def plot_http_distributions_with_range(self, output_path='./'):
        """
        Plot and save histograms showing distributions of HTTP functions
        with Q1, median, and Q3 threshold lines
        """
        import os
        
        print("\nGenerating HTTP function distribution plots...")
        
        # Filter to HTTP functions only
        http_funcs = self.metrics_df[
            (self.metrics_df['Trigger'] == 'http') &
            (self.metrics_df['TotalInvocations'] > 50)
        ].copy()
        
        if len(http_funcs) == 0:
            print("  ⚠ No HTTP functions found!")
            return
        
        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('HTTP Functions Distribution with Q1, Median, Q3 Thresholds', 
                    fontsize=16, fontweight='bold')
        
        # ============================
        # 1. Invocations per Minute (TotalInvocations)
        # ============================
        # ============================
        # 1. Invocations per Minute (TotalInvocations) with log scale
        # ============================
        ax = axes[0]
        
        # Quantiles
        inv_q1 = http_funcs['TotalInvocations'].quantile(0.25)
        inv_median = http_funcs['TotalInvocations'].quantile(0.50)
        inv_q3 = http_funcs['TotalInvocations'].quantile(0.75)
        
        # Use log scale to better visualize skewed distribution
        data = http_funcs['TotalInvocations']
        
        # Histogram with log scale
        bins = np.logspace(np.log10(data.min()), np.log10(data.max()), 50)
        ax.hist(data, bins=bins, alpha=0.7, color='steelblue', edgecolor='black')
        
        # Quantile lines (also plotted on log scale)
        ax.axvline(inv_q1, color='blue', linestyle=':', linewidth=2, label=f'Q1: {inv_q1:.0f}')
        ax.axvline(inv_median, color='red', linestyle='--', linewidth=2, label=f'Median: {inv_median:.0f}')
        ax.axvline(inv_q3, color='green', linestyle=':', linewidth=2, label=f'Q3: {inv_q3:.0f}')
        
        ax.set_xscale('log')
        ax.set_xlabel('Mean Invocations per Minute (log scale)', fontsize=12)
        ax.set_ylabel('Number of Functions', fontsize=12)
        ax.set_title('Invocations Distribution', fontsize=14, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, which='both', alpha=0.3)
        
        # Counts per quartile
        q1_count = (data <= inv_q1).sum()
        med_count = ((data > inv_q1) & (data <= inv_q3)).sum()
        q3_count = (data > inv_q3).sum()
        ax.text(0.02, 0.98, f'Q1: {q1_count}\nMed: {med_count}\nQ3: {q3_count}', 
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
        
        # ============================
        # 2. Duration
        # ============================
        ax = axes[1]
        
        # Calculate quantiles
        dur_q1 = http_funcs['AvgDuration'].quantile(0.25)
        dur_median = http_funcs['AvgDuration'].quantile(0.50)
        dur_q3 = http_funcs['AvgDuration'].quantile(0.75)
        
        durations = http_funcs['AvgDuration']

        # Log-spaced bins
        bins = np.logspace(
            np.log10(durations[durations > 0].min()),
            np.log10(durations.max()),
            40
        )

        ax.hist(
            durations,
            bins=bins,
            color='orange',
            alpha=0.8,
            edgecolor='black',
            linewidth=0.8
        )

        # Quantile lines
        ax.axvline(dur_q1, color='blue', linestyle=':', linewidth=2, label=f'Q1: {dur_q1:.1f} ms')
        ax.axvline(dur_median, color='red', linestyle='--', linewidth=2, label=f'Median: {dur_median:.1f} ms')
        ax.axvline(dur_q3, color='green', linestyle=':', linewidth=2, label=f'Q3: {dur_q3:.1f} ms')

        ax.set_xscale('log')
        ax.set_xlabel('Average Duration (ms, log scale)')
        ax.set_ylabel('Number of Functions')
        ax.set_title('Duration Distribution (Log Scale)', fontweight='bold')

        ax.grid(True, which='both', alpha=0.3)
        ax.legend(fontsize=9)
        
        # Add counts
        q1_count = (http_funcs['AvgDuration'] <= dur_q1).sum()
        med_count = ((http_funcs['AvgDuration'] > dur_q1) & (http_funcs['AvgDuration'] <= dur_q3)).sum()
        q3_count = (http_funcs['AvgDuration'] > dur_q3).sum()
        ax.text(0.02, 0.98, f'Q1: {q1_count}\nMed: {med_count}\nQ3: {q3_count}', 
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # ============================
        # 3. Memory
        # ============================
        ax = axes[2]
        
        if 'EstimatedMemoryMB' in http_funcs.columns:
            # Filter out zero memory
            non_zero_mem = http_funcs[http_funcs['EstimatedMemoryMB'] > 0]['EstimatedMemoryMB']
            
            if len(non_zero_mem) > 0:
                # Calculate quantiles (on non-zero memory)
                mem_q1 = non_zero_mem.quantile(0.25)
                mem_median = non_zero_mem.quantile(0.50)
                mem_q3 = non_zero_mem.quantile(0.75)
                
                # Plot histogram
                ax.hist(non_zero_mem, bins=50, alpha=0.7, 
                        color='green', edgecolor='black')
                
                # Add quantile lines
                ax.axvline(mem_q1, color='blue', linestyle=':', linewidth=2,
                        label=f'Q1: {mem_q1:.1f} MB')
                ax.axvline(mem_median, color='red', linestyle='--', linewidth=2,
                        label=f'Median: {mem_median:.1f} MB')
                ax.axvline(mem_q3, color='green', linestyle=':', linewidth=2,
                        label=f'Q3: {mem_q3:.1f} MB')
                
                # Labels and limits
                ax.set_xlabel('Estimated Memory (MB)', fontsize=12)
                ax.set_ylabel('Number of Functions', fontsize=12)
                ax.set_title('Memory Distribution', fontsize=14, fontweight='bold')
                ax.set_xlim(0, 400)
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3)
                
                # Add counts
                q1_count = (non_zero_mem <= mem_q1).sum()
                med_count = ((non_zero_mem > mem_q1) & (non_zero_mem <= mem_q3)).sum()
                q3_count = (non_zero_mem > mem_q3).sum()
                zero_count = (http_funcs['EstimatedMemoryMB'] == 0).sum()
                ax.text(0.02, 0.98, f'Q1: {q1_count}\nMed: {med_count}\nQ3: {q3_count}\nZero: {zero_count}', 
                        transform=ax.transAxes, fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            else:
                ax.text(0.5, 0.5, 'No memory data available', 
                        transform=ax.transAxes, ha='center', va='center', fontsize=14)
                ax.set_title('Memory Distribution', fontsize=14, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No memory data available', 
                    transform=ax.transAxes, ha='center', va='center', fontsize=14)
            ax.set_title('Memory Distribution', fontsize=14, fontweight='bold')
        
        # ============================
        # Save figure
        # ============================
        plt.tight_layout()
        
        output_file = os.path.join(output_path, 'http_functions_distributions_w_range.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        
        print(f"  ✓ Saved distribution plot to: {output_file}")
        
        # Print summary statistics
        print(f"\n{'='*60}")
        print("HTTP FUNCTIONS SUMMARY STATISTICS")
        print(f"{'='*60}")
        print(f"Total HTTP functions (>50 invocations): {len(http_funcs)}")
        print(f"\nInvocations (Mean Rate per Minute):")
        print(f"  Min: {http_funcs['TotalInvocations'].min():.4f}")
        print(f"  Q1 (25th): {inv_q1:.4f}")
        print(f"  Median: {inv_median:.4f}")
        print(f"  Q3 (75th): {inv_q3:.4f}")
        print(f"  Max: {http_funcs['TotalInvocations'].max():.4f}")
        print(f"\nDuration (ms):")
        print(f"  Min: {http_funcs['AvgDuration'].min():.1f}")
        print(f"  Q1 (25th): {dur_q1:.1f}")
        print(f"  Median: {dur_median:.1f}")
        print(f"  Q3 (75th): {dur_q3:.1f}")
        print(f"  Max: {http_funcs['AvgDuration'].max():.1f}")
        
        if 'EstimatedMemoryMB' in http_funcs.columns and len(non_zero_mem) > 0:
            print(f"\nMemory (MB):")
            print(f"  Min: {non_zero_mem.min():.2f}")
            print(f"  Q1 (25th): {mem_q1:.2f}")
            print(f"  Median: {mem_median:.2f}")
            print(f"  Q3 (75th): {mem_q3:.2f}")
            print(f"  Max: {non_zero_mem.max():.2f}")
            print(f"  Functions with zero memory: {(http_funcs['EstimatedMemoryMB'] == 0).sum()}")
        print(f"{'='*60}\n")
        
        return fig

    def count_functions_by_pattern(self):
        """
        Count and print the number of functions by pattern type
        """
        if self.metrics_df is None:
            print("Metrics data not computed yet!")
            return
        
        pattern_counts = self.metrics_df['Pattern'].value_counts()
        
        print("\nFunction Counts by Pattern:")
        for pattern, count in pattern_counts.items():
            print(f"  {pattern}: {count} functions")
        #save it in a csv
        pattern_counts.to_csv('function_counts_by_pattern.csv', header=['Count'])


        return pattern_counts
    
    def count_functions_by_cluster(self):
        """
        Count and print the number of functions by pattern type,
        and for bursty functions, show counts for each of the 8 clusters
        (Memory H/L × Duration H/L × Frequency H/L)
        """
        if self.metrics_df is None:
            print("Metrics data not computed yet!")
            return
        
        pattern_counts = self.metrics_df['Pattern'].value_counts()
        
        print("\nFunction Counts by Pattern:")
        for pattern, count in pattern_counts.items():
            print(f"  {pattern}: {count} functions")
        
        # Additional breakdown for bursty functions
        bursty_funcs = self.metrics_df[self.metrics_df['Pattern'] == 'bursty']
        
        if len(bursty_funcs) > 0 and 'MemoryCategory' in bursty_funcs.columns:
            print("\nBURSTY Function Breakdown (8 clusters):")
            print("-" * 70)
            
            clusters = [
                ('High', 'High', 'High', 'High Mem + High Dur + High Inv'),
                ('High', 'High', 'Low', 'High Mem + High Dur + Low Inv'),
                ('High', 'Low', 'High', 'High Mem + Low Dur + High Inv'),
                ('High', 'Low', 'Low', 'High Mem + Low Dur + Low Inv'),
                ('Low', 'High', 'High', 'Low Mem + High Dur + High Inv'),
                ('Low', 'High', 'Low', 'Low Mem + High Dur + Low Inv'),
                ('Low', 'Low', 'High', 'Low Mem + Low Dur + High Inv'),
                ('Low', 'Low', 'Low', 'Low Mem + Low Dur + Low Inv'),
            ]
            
            cluster_counts = []
            
            for mem, dur, inv, label in clusters:
                count = len(bursty_funcs[
                    (bursty_funcs['MemoryCategory'] == mem) &
                    (bursty_funcs['DurationCategory'] == dur) &
                    (bursty_funcs['FrequencyCategory'] == inv)
                ])
                print(f"  {label}: {count} functions")
                cluster_counts.append({'Cluster': label, 'Memory': mem, 'Duration': dur, 'Frequency': inv, 'Count': count})
            
            # Save cluster breakdown
            cluster_df = pd.DataFrame(cluster_counts)
            cluster_df.to_csv('bursty_function_clusters.csv', index=False)
            print("\n  ✓ Saved cluster breakdown to bursty_function_clusters.csv")
        
        # Save pattern counts
        pattern_counts.to_csv('function_counts_by_cluster.csv', header=['Count'])
        print("  ✓ Saved pattern counts to function_counts_by_pattern.csv")
        
        return pattern_counts
    

if __name__ == "__main__":
    import os
    
    # Initialize selector
    selector = FunctionSelector(data_path='./data/')
    
    # Load 3 days of data
    print("Loading data...")
    selector.load_data(start_day=1, end_day=9)
    
    # Check for metrics cache
    cache_file = './metrics_cache.csv'
    
    if os.path.exists(cache_file):
        print(f"\nLoading metrics from cache: {cache_file}")
        selector.metrics_df = pd.read_csv(cache_file)
        print(f"Loaded {len(selector.metrics_df)} unique functions from cache")
    else:
        print("\nComputing function metrics...")
        metrics = selector.compute_function_metrics()
        print(f"Analyzed {len(metrics)} unique functions")
        
        # Save to cache
        print(f"Saving metrics cache to: {cache_file}")
        selector.metrics_df.to_csv(cache_file, index=False)
        print("✓ Cache saved")
    
    # Compute memory
    print("\nComputing per-function memory...")
    selector.compute_function_memory()
    
    # Categorize
    print("\nCategorizing functions...")
    selector.categorize_functions()
    selector.categorize_memory()

    print("\n" + "="*70)
    print("Visualizing HTTP function distributions...")
    print("="*70)
    selector.plot_http_distributions()
    
    # Select diverse set (9 bursty + 1 periodic)
    print("\n" + "="*70)
    print("SELECTION STRATEGY: 9 Bursty + 1 Periodic")
    print("="*70)
    selected = selector.select_diverse_functions_rand(
        n_functions=20, 
        http_only=True,
        include_memory=True,
        random_seed=27
    )
   # Print selection matrix
    selector.print_selection_matrix()
    
    print("\n=== SELECTED FUNCTIONS DETAILS ===")
    display_cols = ['HashFunction', 'Trigger', 'Pattern', 'MemoryCategory', 
                    'DurationCategory', 'TotalInvocations', 'AvgDuration', 
                    'EstimatedMemoryMB', 'CV', 'IdlePercent']
    # Filter to only columns that exist
    display_cols = [c for c in display_cols if c in selected.columns]
    print(selected[display_cols])
    
    # Visualize
    print("\nGenerating visualizations...")
    fig = selector.visualize_selected_functions()
    plt.savefig('selected_functions_overview.png', dpi=300, bbox_inches='tight')
    print("Saved visualization to selected_functions_overview.png")
    
    # **ADD THIS SECTION HERE** ⬇️⬇️⬇️
    # Compute percentile data
    print("\n" + "="*70)
    print("Computing percentile data for export...")
    print("="*70)
    selector.compute_percentile_data()
    # **END OF NEW SECTION** ⬆️⬆️⬆️
    
    # Export
    print("\nExporting data...")
    selector.export_selected_data()
    print("\nDone!")
