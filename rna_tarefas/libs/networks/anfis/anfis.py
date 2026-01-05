import numpy as np
from itertools import product
from .fuzzy_membership import (
    gaussian_membership,
    bell_membership,
    sigmoid_membership,
    triangular_membership,
    trapezoidal_membership
)

class ANFIS:
    def __init__(self, n_inputs, n_mfs, mf_type='gaussian'):
        """
        Rede ANFIS (Adaptive Neuro-Fuzzy Inference System).

        Parâmetros
        ----------
        n_inputs : int
            Número de variáveis de entrada
        n_mfs : int
            Número de funções de pertinência por entrada
        mf_type : str
            Tipo de função de pertinência
        """
        self.n_inputs = n_inputs            # Número de variáveis de entrada
        self.n_mfs = n_mfs                  # Número de funções de pertinência por entrada
        self.n_rules = n_mfs ** n_inputs    # Número de regras fuzzy
        self.mf_type = mf_type

        # Selecionar função de pertinência
        self.mf, self.params_per_mf = self._select_membership_function(mf_type)

        # Parâmetros das funções de pertinência
        self.mf_params = np.zeros((n_inputs, n_mfs, self.params_per_mf))

        # Inicializar parâmetros com valores adequados (Grid Partitioning)
        self._initialize_mf_params(mf_type)

        # Parâmetros dos consequentes (regras)
        self.consequents = np.random.randn(self.n_rules, n_inputs + 1) * 0.01

        # Combinações cartesianas para regras
        self.rule_combinations = list(product(range(n_mfs), repeat=n_inputs))

        # Pré-computar máscaras para regras
        self._precompute_rule_masks()


    def _select_membership_function(self, mf_type):
        """Seleciona função de pertinência e retorna número de parâmetros."""
        match mf_type:
            case 'gaussian':
                return gaussian_membership, 2
            case 'bell':
                return bell_membership, 3
            case 'sigmoid':
                return sigmoid_membership, 2
            case 'triangular':
                return triangular_membership, 3
            case 'trapezoidal':
                return trapezoidal_membership, 4
            case _:
                raise ValueError(f"Função de pertinência desconhecida: {mf_type}")


    def _initialize_mf_params(self, mf_type):
        """Inicializa com Grid Partitioning (espaçamento uniforme)."""
        if mf_type == 'gaussian':
            if self.n_mfs == 1:
                centers = np.array([0.5])
                sigma = 0.5
            else:
                centers = np.linspace(0, 1, self.n_mfs)
                sigma = 1.0 / (self.n_mfs - 1) # Desvio padrão baseado no espaçamento

            for i in range(self.n_inputs):
                self.mf_params[i, :, 0] = centers  # Centros
                self.mf_params[i, :, 1] = sigma    # Desvios padrão

        else:
            # Inicialização aleatória para outras funções
            self.mf_params = np.random.rand(self.n_inputs, self.n_mfs, self.params_per_mf)


    def _fuzzify(self, X):
        """Camada 1: Fuzzificação dos dados de entrada."""
        n_samples = X.shape[0]
        fuzzified = np.zeros((n_samples, self.n_inputs, self.n_mfs))

        for i in range(self.n_inputs):
            for j in range(self.n_mfs):
                params = self.mf_params[i, j]
                fuzzified[:, i, j] = self.mf(X[:, i], *params)

        return fuzzified


    def _rules_evaluation(self, fuzzified):
        """Camada 2: Avaliação das regras fuzzy (produto T-norma)."""
        n_samples = fuzzified.shape[0]
        rule_activations = np.zeros((n_samples, self.n_rules))

        # Calcular ativação de cada regra
        for r, combo in enumerate(self.rule_combinations):
            # combo = [(0, 0), (0, 1), ...] índices das MFs para cada input
            activation = np.ones(n_samples)
            for i, mf_idx in enumerate(combo):
                # Produto das pertinências
                activation *= fuzzified[:, i, mf_idx] # Multiplicação elemento a elemento
            rule_activations[:, r] = activation

        return rule_activations


    def _normalize_activations(self, rule_activations):
        """Camada 3: Normalização das ativações."""
        sum_activations = np.sum(rule_activations, axis=1, keepdims=True)
        sum_activations = np.where(sum_activations == 0, 1e-10, sum_activations)
        return rule_activations / sum_activations


    def _defuzzify(self, normalized_activations, X):
        """Camadas 4 e 5: Consequentes e Saída (Vetorizado)."""
        n_samples = X.shape[0]
        # X_bias: (n_samples, n_inputs + 1)
        X_bias = np.column_stack([X, np.ones(n_samples)])

        # Saídas lineares das regras: (n_samples, n_rules)
        linear_outputs = X_bias @ self.consequents.T # X W^T

        # Saída final: soma ponderada das saídas lineares
        y_hat = np.sum(normalized_activations * linear_outputs, axis=1)

        return y_hat


    def forward(self, X):
        """Propagação completa no ANFIS."""
        fuzzified = self._fuzzify(X)
        rule_activations = self._rules_evaluation(fuzzified)
        normalized_activations = self._normalize_activations(rule_activations)
        y_hat = self._defuzzify(normalized_activations, X)
        return y_hat


    def _compute_jacobian_matrix(self, X, y_pred, fuzzified, rule_activations):
        """
        Calcula o Jacobiano das saídas em relação aos parâmetros das MFs
        """
        n_samples = X.shape[0]
        n_params = self.n_inputs * self.n_mfs * 2       # 2 parâmetros por MF (c e σ)
        J = np.zeros((n_samples, n_params))

        # Calcular termos comuns fora do loop
        X_bias = np.column_stack([X, np.ones(n_samples)])
        f_r = X_bias @ self.consequents.T # (N, n_rules)

        # Soma das ativações das regras (N, 1)
        sum_w = np.sum(rule_activations, axis=1, keepdims=True) + 1e-12

        #  (y_r - y_pred) / sum_w
        # Derivada da saída em relação à ativação da regra r
        dy_wk = (f_r - y_pred[:, np.newaxis]) / sum_w

        # Loop sobre cada parâmetro das MFs
        param_idx = 0

        for i in range(self.n_inputs):
            for j in range(self.n_mfs):
                # Parâmetros atuais da Gaussiana
                c = self.mf_params[i, j, 0]
                s = self.mf_params[i, j, 1]

                # Derivadas da Gaussiana em relação a c e sigma
                # mu_ij: (N,)
                mu_ij = fuzzified[:, i, j]

                s_safe = s if s > 1e-6 else 1e-6 # Evitar div por zero

                # d(mu)/dc = mu * (x - c) / s^2
                dmu_dc = mu_ij * (X[:, i] - c) / (s_safe**2)

                # d(mu)/ds = mu * (x - c)^2 / s^3
                dmu_ds = mu_ij * ((X[:, i] - c)**2) / (s_safe**3)

                # Identificar regras que usam esta MF
                relevant_mask = self.rule_masks[(i, j)]

                # Se nenhuma regra usa esta MF, pular
                if not np.any(relevant_mask):
                    param_idx += 2
                    continue

                # Extrair ativações e contribuições relevantes
                w_relevant = rule_activations[:, relevant_mask]
                activation_contributions = dy_wk[:, relevant_mask]

                # Divisão segura por mu_ij
                mu_ij_safe = np.where(mu_ij < 1e-10, 1e-10, mu_ij)[:, np.newaxis]

                # Cálculo de dw/dmu
                dw_dmu = w_relevant / mu_ij_safe

                # Cálculo de dy/dmu
                dy_dmu = np.sum(activation_contributions * dw_dmu, axis=1)

                # Preencher Jacobiano
                J[:, param_idx] = -dy_dmu * dmu_dc      # Para c
                J[:, param_idx + 1] = -dy_dmu * dmu_ds  # Para sigma

                param_idx += 2

        return J


    def _precompute_rule_masks(self):
        """
        Pré-computa máscaras booleanas para identificar quais regras
        usam cada função de pertinência (i, j).
        """
        self.rule_masks = {}
        for i in range(self.n_inputs): # x1, x2
            for j in range(self.n_mfs): # MF1, MF2
                # Máscara: quais regras usam MF j no input i
                mask = np.array([combo[i] == j for combo in self.rule_combinations])
                self.rule_masks[(i, j)] = mask


    def train(self, X, y, n_epochs=100, tolerance=1e-5, alpha=0.01,
          alpha_variability=10, sigma_min=1e-4, stopping_criteria=[1, 2, 3], verbose=True):
        """
        Treinamento Híbrido ANFIS com Levenberg-Marquardt otimizado.

        Parâmetros
        ----------
        X : np.ndarray, shape (n_samples, n_inputs)
            Dados de entrada
        y : np.ndarray, shape (n_samples,)
            Valores alvo
        n_epochs : int
            Número máximo de épocas
        tolerance : float
            Tolerância para critérios de parada
        alpha : float
            Lambda inicial do Levenberg-Marquardt
        alpha_variability : float
            Fator de ajuste do lambda
        sigma_min : float
            Valor mínimo permitido para σ (evita colapso das Gaussianas)
        stopping_criteria : list of int
            Lista com critérios de parada ativos:
            - 1: Parar se ||gradiente|| < tolerance
            - 2: Parar se ||Δparams|| < tolerance
            - 3: Parar se |ΔMSE| < tolerance
        verbose : bool
            Mostrar logs de progresso

        Retorna
        -------
        history : list
            Histórico de MSE por época
        n_epochs_executed : int
            Número de épocas executadas
        """
        n_samples = X.shape[0]
        X_bias = np.column_stack([X, np.ones(n_samples)])

        lambda_param = alpha
        history = []
        params_history = []
        n_epochs_executed = 0

        for epoch in range(n_epochs):
            n_epochs_executed = epoch + 1

            # Atualização dos consequentes via LSE
            fuzzified = self._fuzzify(X)
            rule_activations = self._rules_evaluation(fuzzified)
            normalized = self._normalize_activations(rule_activations)

            # Construir matriz A para LSE
            A_elements = normalized[:, :, np.newaxis] * X_bias[:, np.newaxis, :]
            A = A_elements.reshape(n_samples, -1)
            # A = w_ij * X_bias para todas as regras concatenadas

            # Resolver LSE
            try:
                theta, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
                self.consequents = theta.reshape(self.n_rules, self.n_inputs + 1)
            except np.linalg.LinAlgError:
                if verbose:
                    print(f"Epoch {epoch}: LSE singular. Pulando atualização dos consequentes.")

            # Calcular predição e MSE
            y_pred = self.forward(X)
            residuals = y - y_pred
            current_mse = np.mean(residuals**2)
            history.append(current_mse)

            # Log de progresso
            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch:4d}/{n_epochs}: MSE = {current_mse:.6e}, λ = {lambda_param:.2e}")

            # Calcular Jacobiano
            J = self._compute_jacobian_matrix(X, y_pred, fuzzified, rule_activations)

            # Gradiente e Hessiana aproximada
            gradient = -J.T @ residuals
            H_approx = J.T @ J
            identity_matrix = np.eye(H_approx.shape[0])

            # Critério de parada 1: Norma do gradiente
            if 1 in stopping_criteria and np.abs(gradient).max() <= tolerance:
                if verbose:
                    print(f"Convergiu na época {epoch} pelo critério da norma do gradiente (||grad|| = {np.abs(gradient).max():.2e})")
                break

            # Atualização dos parâmetros via Levenberg-Marquardt
            try:
                delta = np.linalg.solve(H_approx + lambda_param * identity_matrix, -gradient)
            except np.linalg.LinAlgError:
                if verbose:
                    print(f"Epoch {epoch}: LM singular. Fallback.")
                delta = 0.01 * gradient

            # Critério de parada 2: Mudança nos parâmetros
            if 2 in stopping_criteria and len(params_history) > 0:
                params_change = np.abs(delta).max()
                if params_change < tolerance:
                    if verbose:
                        print(f"Convergiu na época {epoch} pelo critério da mudança nos parâmetros (||Δparams|| = {params_change:.2e})")
                    break

            # Atualizar parâmetros das MFs
            old_params = self.mf_params.copy()

            # Aplicar atualização
            flat_params = self.mf_params.flatten()
            new_flat_params = flat_params - delta
            self.mf_params = new_flat_params.reshape(self.mf_params.shape)

            # Salvar histórico de parâmetros
            params_history.append(self.mf_params.copy())

            # Regularização de σ
            self.mf_params[:, :, 1] = np.maximum(self.mf_params[:, :, 1], sigma_min)

            new_pred = self.forward(X)
            new_mse = np.mean((y - new_pred)**2)

            # Critério de parada 3: Mudança no MSE
            if 3 in stopping_criteria and len(history) > 1:
                mse_change = np.abs(current_mse - history[-2])
                if mse_change < tolerance:
                    if verbose:
                        print(f"Convergiu na época {epoch} pelo critério da mudança no MSE (|ΔMSE| = {mse_change:.2e})")
                    break

            # Se o MSE diminuiu, Gauss Netwon
            if new_mse < current_mse:
                lambda_param /= alpha_variability
            # Se o MSE não diminuiu, aumentar lambda e reverter parâmetros
            else:
                self.mf_params = old_params
                lambda_param *= alpha_variability

        return history, n_epochs_executed
