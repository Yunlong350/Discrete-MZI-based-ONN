import numpy as np
import time
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from scipy.fftpack import dct

class MZIMeshBase:
    def __init__(self, 
                 input_size,        
                 mesh_type,         
                 output_size, 
                 batch_size, 
                 num_layer, 
                 seed=1,             
                 noise_seed=42,      
                 optimizer='adam',         
                 enable_phase_noise=False, 
                 phase_noise_std=0, 
                 n_noise_bins=int(1e5),
                 enable_bs_noise=False,
                 bs_noise_std=0
                 ): 
        
        self.rng = np.random.RandomState(seed)           
        self.noise_rng = np.random.RandomState(noise_seed) 
        
        self.real_input_size = input_size 
        if input_size % 2 != 0:
            self.mesh_input_size = input_size + 1 
            self.pad_input = True
        else:
            self.mesh_input_size = input_size
            self.pad_input = False

        self.mesh_type = mesh_type
        
        if self.mesh_type == 'Reck':
            self.mesh = self._build_reck_mesh(self.mesh_input_size)
        elif self.mesh_type == 'Clements':
            self.mesh = self._build_clements_mesh(self.mesh_input_size)
        else:
            raise ValueError(f"Unknown mesh type: {self.mesh_type}")
            
        self.output_size = output_size
        self.batch_size = batch_size
        self.num_layer = num_layer
        self.optimizer = optimizer
        
        self.n_mzi = np.sum(self.mesh)
        self.n_phase_mesh = num_layer * self.n_mzi * 2
        
        self.W_out = np.zeros((self.mesh_input_size, output_size))
        self.b_out = np.zeros((1, output_size))
        for i in range(self.mesh_input_size - self.mesh_input_size % output_size):
            target_class = i % output_size 
            self.W_out[i, target_class] = 1.0
    
        self.phase = None
        self.m_phase = None 
        self.v_phase = None
        
        self.beta1 = 0.9  
        self.beta2 = 0.999  
        self.epsilon = 1e-8  
        
        self.enable_phase_noise = enable_phase_noise
        self.enable_bs_noise = enable_bs_noise
        self.phase_noise_std = phase_noise_std
        self.n_noise_bins = n_noise_bins
        self.bs_noise_std = bs_noise_std 
        
        self.phase_noise_map = None
        self.bs_noise = None
        
        self._initialize_hardware_noise()
        self._initialize_params()

    def _build_reck_mesh(self, n_inputs):
        w = n_inputs - 1
        l = 2 * w - 1
        mesh = np.zeros((w, l), dtype=np.int32)
        for i in range(w):
            m = [1, 0] * (w - 1 - i) + [1]
            mesh[w - 1 - i][i:l - i] = m
        return mesh

    def _build_clements_mesh(self, n_inputs):
        l = n_inputs
        w = l - 1
        mesh = np.zeros((w, l), dtype=np.int32)
        c1 = [1, 0] * int(l / 2 - 1) + [1]
        c2 = [0, 1] * int(l / 2 - 1) + [0]
        for j in range(l):
            if j % 2 == 0:
                mesh[:, j] = c1
            else:
                mesh[:, j] = c2
        return mesh

    def _initialize_hardware_noise(self):
        if self.enable_phase_noise:
            self.phase_noise_map = self.noise_rng.normal(
                loc=0.0, scale=self.phase_noise_std, size=(self.n_phase_mesh, self.n_noise_bins))
        else:
            self.phase_noise_map = None

        if self.enable_bs_noise:
            self.bs_noise = self.noise_rng.normal(0, self.bs_noise_std, self.n_phase_mesh)
        else:
            self.bs_noise = None

    def _initialize_params(self):
        self.phase = self.rng.uniform(0, 2 * np.pi, self.n_phase_mesh)
        self.m_phase = np.zeros_like(self.phase)
        self.v_phase = np.zeros_like(self.phase)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def quantize_phase(self):
        pass
    
    def MZI(self, theta, phi, alpha=0, beta=0):
        M = np.array([
            [np.exp(1j * phi) * np.cos(theta), np.sin(theta)],
            [np.exp(1j * phi) * np.sin(theta), -np.cos(theta)]
        ])
        if self.enable_bs_noise:
            noise_beta = np.array([[np.cos(beta), 1j*np.sin(beta)],
                                   [1j*np.sin(beta), np.cos(beta)]])
            noise_alpha = np.array([[np.cos(alpha), 1j*np.exp(-1j*phi)*np.sin(alpha)],
                                    [1j*np.exp(1j*phi)*np.sin(alpha), np.cos(alpha)]])
            M = noise_beta.dot(M).dot(noise_alpha)
        return M

    def encode_input(self, X_real):
        target_phase = 2 * np.pi * X_real
        batch_size = X_real.shape[0]
        X_complex = np.zeros((batch_size, self.mesh_input_size), dtype=np.complex128)
        X_complex[:, :self.real_input_size] = np.exp(1j * target_phase)
        return X_complex

    def optical_layer(self, X0, layer_idx):
        current_phi_ideal = self.phase
        start_idx = layer_idx * self.n_mzi * 2
        end_idx = start_idx + 2 * self.n_mzi
        if self.enable_phase_noise and self.phase_noise_map is not None:
            phase_mod = np.mod(current_phi_ideal, 2 * np.pi)
            bin_indices = (phase_mod / (2 * np.pi) * (self.n_noise_bins - 1)).astype(int)
            current_noise = self.phase_noise_map[np.arange(self.n_phase_mesh), bin_indices]
            noisy_phase_total = current_phi_ideal + current_noise
        else:
            noisy_phase_total = current_phi_ideal
            
        layer_phases = noisy_phase_total[start_idx:end_idx]
        theta = layer_phases[0:self.n_mzi]
        phi = layer_phases[self.n_mzi:]
        if self.enable_bs_noise and self.bs_noise is not None:
            alpha = self.bs_noise[start_idx:start_idx+self.n_mzi]
            beta = self.bs_noise[start_idx+self.n_mzi:start_idx+2*self.n_mzi]
        else:
            alpha, beta = np.zeros_like(theta), np.zeros_like(phi)
        X1 = np.array(X0, dtype=np.complex128)
        k = 0
        for i in range(self.mesh.shape[1]):
            for j in range(self.mesh.shape[0]):
                if self.mesh[j][i] == 1:
                    M = self.MZI(theta[k], phi[k], alpha[k], beta[k])
                    X1[:, j:j+2] = X1[:, j:j+2].dot(M.T)
                    k += 1    
        return X1
    
    def forward(self, X_real):
        X_optical = self.encode_input(X_real)
        for layer in range(self.num_layer):
            X_optical = self.optical_layer(X_optical, layer)
        intensity = np.abs(X_optical)**2
        a_out, z_out = self.fc_forward(intensity)
        return a_out, z_out, intensity
    
    def fc_forward(self, intensity):
        z_out = np.dot(intensity, self.W_out) + self.b_out
        return self.softmax(z_out), z_out
    
    def compute_loss(self, a_out, y_true):
        return np.sum(-np.log(a_out + 1e-8) * y_true) / a_out.shape[0]

    def SPSA(self, X0, y_true, iterations, a=0.1, d=10):

        c_k = 2*np.pi/2**6#0.01 #
        a_k = a / (iterations + 1 + d)**0.602
        
        delta = self.rng.choice([-1, 1], self.n_phase_mesh)
        p0 = self.phase.copy() 
        self.phase = p0 + c_k * delta
        a_out_plus, _, _ = self.forward(X0)
        loss_plus = self.compute_loss(a_out_plus, y_true)
        self.phase = p0 - c_k * delta
        a_out_minus, _, _ = self.forward(X0)
        loss_minus = self.compute_loss(a_out_minus, y_true)
        self.phase = p0
        grad_mesh = (loss_plus - loss_minus) / (2 * c_k * delta)
        return grad_mesh, a_k

    def train(self, X, y, X_test, y_test, epochs, verbose=True):
        losses = np.zeros(epochs + 1)
        train_accuracy = np.zeros(epochs + 1)
        test_accuracy = np.zeros(epochs + 1)
        a_out_init, _, _ = self.forward(X)
        losses[0] = self.compute_loss(a_out_init, y)
        train_accuracy[0] = self.accuracy(X, y)
        test_accuracy[0] = self.accuracy(X_test, y_test)
        if verbose:
            print(f"Epoch 0 (Init), Loss {losses[0]:.4f}, Train Acc {train_accuracy[0]:.4f}, Test Acc{test_accuracy[0]:.4f}")

        current_iteration = 0
        for epoch in range(1, epochs + 1):
            indices = self.rng.permutation(X.shape[0])
            X_shuffled, y_shuffled = X[indices], y[indices]
            total_loss = 0
            
            for i in range(0, X.shape[0], self.batch_size):
                X_batch, y_batch = X_shuffled[i:i+self.batch_size], y_shuffled[i:i+self.batch_size]
                a_out_batch, _, _ = self.forward(X_batch)
                total_loss += self.compute_loss(a_out_batch, y_batch) * X_batch.shape[0]
                grad_mesh, a_k = self.SPSA(X_batch, y_batch, current_iteration)
                self.update_params(grad_mesh, current_iteration, a_k)
                current_iteration += 1
                
            losses[epoch] = total_loss / X.shape[0]
            train_accuracy[epoch] = self.accuracy(X, y)
            test_accuracy[epoch] = self.accuracy(X_test, y_test)
            if verbose and (epoch % 1 == 0 or epoch < 10):
                print(f"Epoch {epoch}, Loss {losses[epoch]:.4f}, Train Acc {train_accuracy[epoch]:.4f}, Test Acc{test_accuracy[epoch]:.4f}")
        
        return losses, train_accuracy, test_accuracy
    
    def predict(self, X): 
        
        return np.argmax(self.forward(X)[0], axis=1)
    def accuracy(self, X, y_true): 
        
        return np.mean(self.predict(X) == np.argmax(y_true, axis=1))
    
    def update_params(self, grad_mesh, iterations, eta_mesh): 
        
        pass

class MZIMeshContinuous(MZIMeshBase):
    def __init__(self, **kwargs): 
        super().__init__(**kwargs) 
        self.is_discrete = False
    
    def update_params(self, grad_mesh, iterations, eta_mesh):
        if self.optimizer == 'adam':
            self.m_phase = self.beta1 * self.m_phase + (1 - self.beta1) * grad_mesh
            self.v_phase = self.beta2 * self.v_phase + (1 - self.beta2) * (grad_mesh ** 2)
            m_hat = self.m_phase / (1 - self.beta1 ** (iterations+1))
            v_hat = self.v_phase / (1 - self.beta2 ** (iterations+1))
            self.phase -= eta_mesh * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return 0

class MZIMeshDiscrete(MZIMeshBase):
    def __init__(self, input_size, mesh_type, output_size, batch_size, num_layer, discrete_level, 
                 seed=1, noise_seed=42, optimizer='adam', 
                 enable_phase_noise=False, phase_noise_std=0, n_noise_bins=int(1e5),
                 enable_bs_noise=False, bs_noise_std=0): 
        
        self.discrete_level = discrete_level
        super().__init__(input_size, mesh_type, output_size, batch_size, num_layer, 
                         seed, noise_seed, optimizer, 
                         enable_phase_noise, phase_noise_std, n_noise_bins, enable_bs_noise, bs_noise_std) 
        

        Q = int(np.log2(self.discrete_level))
        B = Q + 1
        n_dac = 2**B
        v_dac = np.arange(n_dac+1) / n_dac 
        phi_dac_all = 2 * np.pi * (v_dac**2)
        
        
        phi_targets = np.linspace(0, 2 * np.pi, self.discrete_level+1, endpoint=True)
        self.lut_phases = np.zeros_like(phi_targets)
        for i,t in enumerate(phi_targets):
            idx = np.argmin(np.abs(phi_dac_all - t))
            self.lut_phases[i]=phi_dac_all[idx]
        
        #self.lut_phases=phi_dac_all
        
        self.phase_continuous = self.phase.copy()
        self.phase_indices = np.zeros(self.n_phase_mesh, dtype=int) # 新增：记录索引
        self.is_discrete = True
        self.quantize_phase()
        
    def quantize_phase(self):
        phase_wrapped = np.mod(self.phase_continuous, 2 * np.pi)
        self.phase_indices = np.argmin(np.abs(phase_wrapped[..., None] - self.lut_phases), axis=-1)
        self.phase = self.lut_phases[self.phase_indices]

    def SPSA(self, X0, y_true, iterations, a=0.1, d=10):

        a_k = a / (iterations + 1 + d)**0.602
        delta = self.rng.choice([-1, 1], self.n_phase_mesh)
        
        p0 = self.phase.copy()
        idx0 = self.phase_indices.copy()

        idx_plus = np.mod(idx0 + delta, self.discrete_level)
        self.phase = self.lut_phases[idx_plus]
        a_out_plus, _, _ = self.forward(X0)
        loss_plus = self.compute_loss(a_out_plus, y_true)
        

        idx_minus = np.mod(idx0 - delta, self.discrete_level)
        self.phase = self.lut_phases[idx_minus]
        a_out_minus, _, _ = self.forward(X0)
        loss_minus = self.compute_loss(a_out_minus, y_true)
        
        self.phase = p0
        

        phi_diff = self.lut_phases[idx_plus] - self.lut_phases[idx_minus]
 
        
        grad_mesh = (loss_plus - loss_minus) / (phi_diff + 1e-12)
        return grad_mesh, a_k

    def update_params(self, grad_mesh, iterations, eta_mesh):
        if self.optimizer == 'adam':
            self.m_phase = self.beta1 * self.m_phase + (1 - self.beta1) * grad_mesh
            self.v_phase = self.beta2 * self.v_phase + (1 - self.beta2) * (grad_mesh ** 2)
            m_hat = self.m_phase / (1 - self.beta1 ** (iterations+1))
            v_hat = self.v_phase / (1 - self.beta2 ** (iterations+1))
            self.phase_continuous -= eta_mesh * m_hat / (np.sqrt(v_hat) + self.epsilon)
        self.quantize_phase()
        return 0


def dct2(a): 
    
    return dct(dct(a.T, norm='ortho').T, norm='ortho')
def get_zigzag_indices(r, c, n):
    idx = [(i, j) for i in range(r) for j in range(c)]
    idx.sort(key=lambda x: (x[0] + x[1], x[1] if (x[0] + x[1]) % 2 else -x[1]))
    return idx[:n]

def perform_dct_reduction(X, n):
    N, h, w = X.shape
    zigzag = get_zigzag_indices(h, w, n) 
    X_red = np.zeros((N, n))
    for i in range(N):
        coef = dct2(X[i]) 
        X_red[i] = np.array([coef[r, c] for r, c in zigzag])
    
    return X_red


if __name__ == "__main__":
    t1 = time.perf_counter()
    train_seed = 1      
    hardware_seed = 999 
    
    try:
        with np.load('./mnist.npz', allow_pickle=True) as data:
            X_train_original, y_train_original = data['x_train'], data['y_train']
            X_test_original, y_test_original = data['x_test'], data['y_test']
    except FileNotFoundError:
        raise

    if X_train_original.ndim == 2:
        X_train_img = X_train_original.reshape(-1, 28, 28)
        X_test_img = X_test_original.reshape(-1, 28, 28)
    else:
        X_train_img, X_test_img = X_train_original, X_test_original
    
    n_features=64
    X_train_reduced = perform_dct_reduction(X_train_img, n_features)
    X_test_reduced = perform_dct_reduction(X_test_img, n_features)
    
    std_scaler = StandardScaler()
    X_train_std = std_scaler.fit_transform(X_train_reduced)
    X_test_std = std_scaler.transform(X_test_reduced)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train_std)
    X_test = scaler.transform(X_test_std)     
    
    encoder = OneHotEncoder(sparse_output=False)
    y_train = encoder.fit_transform(y_train_original.reshape(-1, 1))
    y_test = encoder.transform(y_test_original.reshape(-1, 1))

    input_dim = X_train.shape[1]
    output_size = 10
    batch_size = 500
    epochs = 201
    layer = 1

    initial_discrete_level=2**6
    Qbit=7#'inf'#
    
    onn_phase_initialization = MZIMeshDiscrete(
        input_size=input_dim,    
        mesh_type='Clements',    
        output_size=output_size, 
        batch_size=batch_size, 
        num_layer=layer, 
        discrete_level=initial_discrete_level,
        seed=train_seed,           
        noise_seed=hardware_seed,  
        optimizer='adam',
        enable_phase_noise=False,
        phase_noise_std=0,
        n_noise_bins=int(1e2),
        enable_bs_noise=False,
        bs_noise_std=0
    )
    
    if Qbit=='inf':
        onn= MZIMeshContinuous(
            input_size=input_dim,    
            mesh_type='Clements',    
            output_size=output_size, 
            batch_size=batch_size, 
            num_layer=layer, 
            seed=train_seed,           
            noise_seed=hardware_seed,   
            enable_phase_noise=False,
            phase_noise_std=0,
            n_noise_bins=int(1e5),
            enable_bs_noise=False,
            bs_noise_std=0
        )
        
    else:
        onn= MZIMeshDiscrete(
            input_size=input_dim,    
            mesh_type='Clements',    
            output_size=output_size, 
            batch_size=batch_size, 
            num_layer=layer, 
            discrete_level=2**Qbit,
            seed=train_seed,           
            noise_seed=hardware_seed,  
            optimizer='adam',
            enable_phase_noise=False,
            phase_noise_std=0,
            n_noise_bins=int(1e5),
            enable_bs_noise=False,
            bs_noise_std=0
        )
    
    onn.phase=onn_phase_initialization.phase
    losses, train_acc, test_acc = onn.train(X_train, y_train, X_test, y_test, epochs)

    plt.figure()
    plt.plot(range(epochs + 1), losses)
    plt.savefig('loss.pdf')
    
    plt.figure()
    plt.plot(range(epochs + 1), train_acc)
    plt.savefig('acc.pdf')
    
    import xlwings as xw
    file_path = f'./result/n_features={n_features}/Qbit={Qbit}.xlsx'
    try: wb = xw.Book(file_path)
    
    except: wb = xw.Book(); wb.save(file_path)
    
    wb.display_alerts = False; wb.screen_updating = False
    sht=wb.sheets.add('Qbit='+str(Qbit)+'_std='+str(onn.phase_noise_std))
    sht.range('A1').value=['epoch', 'losses', 'training accuracy', 'test accuracy', 'phase']
    sht.range('A2').value = np.column_stack((range(epochs+1), losses, train_acc, test_acc))
    sht.range('E2').value = [[p] for p in onn.phase]
  