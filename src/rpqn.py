import numpy as np
from tqdm import tqdm

def RPQN(config, problem, x0, x_true, tqdm_bar=True):
    mu = 1
    p_min = 1e-4
    mem = config.mem
    method = config.method
    c1 = config.c1
    c2 = config.c2
    sgm1 = config.sgm1
    sgm2 = config.sgm2
    crit_stop = config.crit_stop
    num_iter = config.num_iter

    grad_f = problem.grad_f
    prox = problem.prox
    prox_jacob = problem.prox_jacob
    phi = problem.phi
    psi = problem.psi
    crit = problem.crit

    n = len(x0)

    prev_x = np.copy(x0)
    prev_grad = grad_f(prev_x)
    gamma = 1.
    S = None
    Y = None
    x_hst = []
    cur_iter = 0
    S = np.zeros((n, num_iter))
    Y = np.zeros((n, num_iter))
    j = 1

    crit_list = []
    func_diff_list = []

    succes_iter_flag = True
    x_true_norm = np.linalg.norm(x_true)
    rel_er = np.linalg.norm(prev_x - x_true) / np.max([x_true_norm, 1])
    itr = (tqdm(range(1, num_iter + 1)) if tqdm_bar else range(1, num_iter + 1))
    for j in itr:
        crit_list.append(crit(prev_x))
        func_diff_list.append(psi(prev_x))
        rel_er = np.linalg.norm(prev_x - x_true) / np.max([x_true_norm, 1])
        x_hst.append(rel_er)
        if rel_er < crit_stop:
            break
        if j == 1:
            gamma = np.linalg.norm(prev_grad) / 1.
            #print(f"gamma={gamma}")
            gamma_top = (mu + gamma)
            #gamma_top = 1e10
            x = prox(prev_x - ((1. / gamma_top) * prev_grad).flatten(), gamma_top)
            grad = grad_f(x)

            S[:, cur_iter] = x - prev_x
            Y[:, cur_iter] = grad - prev_grad
            cur_iter += 1
            succes_iter_flag = True

            prev_x = x
            prev_grad = grad
        else:
            s, y = S[:,cur_iter-1], Y[:,cur_iter-1]
            #if (s.T @ y) < eps * (s.T @ s.T):
            gamma = (y.T @ y) / (s.T @ y)
            gamma_top = gamma + mu
            gm_tp_inv = 1. / gamma_top

            if succes_iter_flag == True:
                left_bor = np.max([0, cur_iter - mem])
                Sk = S[:, left_bor:cur_iter]
                Yk = Y[:, left_bor:cur_iter]
                SkT_mul_Yk = Sk.T @ Yk
                D = np.diag(np.diag(SkT_mul_Yk))
                L = np.tril(SkT_mul_Yk, -1)
                if method == 'BFGS':
                    A = np.hstack((gamma * Sk, Yk))
                    Q = np.block([[-gamma * (Sk.T @ Sk), -L], [-L.T, D]])
                elif method == 'SR1':
                    A = Yk - gamma * Sk
                    Q = D + L + L.T - (gamma * Sk.T) @ Sk
                elif method == 'PSB':
                    U = np.triu(SkT_mul_Yk, 0)
                    A = np.hstack((Sk, Yk - gamma * Sk))
                    Q = np.block([[np.zeros_like(U), U], [U.T, L + D + L.T]])
                elif method == "DFP":
                    A = np.hstack((gamma * Sk, Yk))
                    L_top = np.tril(SkT_mul_Yk, 0)
                    Q = -np.block([[(D + (gamma * Sk.T) @ Sk), L_top], [L_top.T, np.zeros_like(L_top)]])

                l, V = np.linalg.eigh(Q)
                l1 = 1. / l
                eigval_eps = 1e-8
                r1, r2 = len(l[l > eigval_eps]), len(l[l < -eigval_eps])
                #print("r1, r2 =", (r1, r2))
                ln = r1 + r2
                #print(f"ln={ln}")
                U1 = (A @ V)[:, l > eigval_eps] * np.sqrt(l1[l > eigval_eps])
                U2 = (A @ V)[:, l < -eigval_eps] * np.sqrt(-l1[l < -eigval_eps])

            Bk1_top_inv_mul_U2 = gm_tp_inv * U2 - (gm_tp_inv ** 2) * (U1 @ (np.linalg.inv(np.identity(r1) + gm_tp_inv * U1.T @ U1) @ (U1.T @ U2)))

            i = 0
            alpha = np.zeros(ln)
            L = None
            Bk1_top_inv_mul_U2_mul_alpha = None
            #print("r1 =", r1, "r2 =", r2)
            #print(alpha.shape)
            alpha1, alpha2 = None, None
            while ((i == 0) or np.linalg.norm(L) > 1e-10) and (i <= 10):
                alpha1 = (alpha[:r1] if r1 != 0 else np.zeros(shape=(0)))
                alpha2 = (alpha[-r2:] if r2 != 0 else np.zeros(shape=(0)))
                #print(alpha.shape, alpha1.shape, alpha2.shape, Bk1_top_inv_mul_U2.shape)
                Bk1_top_inv_mul_U2_mul_alpha = Bk1_top_inv_mul_U2 @ alpha2
                #print(prev_x.shape, Bk1_top_inv_mul_U2_mul_alpha.shape)
                #print("================")
                z = prev_x + Bk1_top_inv_mul_U2_mul_alpha - (gm_tp_inv * (U1 @ alpha1))
                G1 = (np.hstack((U1, U2)).T) @ prox_jacob(z, gamma_top, np.hstack((gm_tp_inv * U1, -Bk1_top_inv_mul_U2)))
                G2 = np.block([[np.identity(r1), U1.T @ Bk1_top_inv_mul_U2], [np.zeros((r2, r1)), np.identity(r2)]])
                G = G1 + G2
                prx = prox(z.flatten(), gamma_top)
                L1 = (U1.T @ (prev_x + (Bk1_top_inv_mul_U2_mul_alpha) - prx) + alpha1).reshape(-1, 1)
                L2 = (U2.T @ (prev_x - prx) + alpha2).reshape(-1, 1)
                L = np.vstack((L1, L2)).reshape(-1)
                alpha = np.hstack((alpha1, alpha2))
                alpha = np.linalg.solve(G, G @ alpha - L)
                i += 1

            prev_grad = grad_f(prev_x)

            Bk1_top_inv_mul_prev_grad = (gm_tp_inv * prev_grad) - (1 / (gamma_top ** 2)) * (U1 @ (np.linalg.inv(np.identity(r1) + gm_tp_inv * U1.T @ U1) @ (U1.T @ prev_grad)))

            B_top_inv_mul_grad = Bk1_top_inv_mul_prev_grad + (Bk1_top_inv_mul_U2 @ np.linalg.inv(np.identity(r2) - (U2.T @ Bk1_top_inv_mul_U2)) @ U2.T) @ (Bk1_top_inv_mul_prev_grad)

            x = prox(prev_x - B_top_inv_mul_grad + Bk1_top_inv_mul_U2_mul_alpha - (gm_tp_inv * U1) @ alpha1, gamma_top)
            grad = grad_f(x)

            d = x - prev_x

            pred = -(prev_grad @ d + phi(x) - phi(prev_x)) - 0.5 * (d @ (gamma * d + U1 @ (U1.T @ d) - U2 @ (U2.T @ d)))

            ared = psi(prev_x) - psi(x)
            ro = ared / pred
            if pred <= p_min * np.linalg.norm(d) * crit(prev_x):
                mu *= sgm2
                succes_iter_flag = False
            elif ro <= c1:
                mu *= sgm2
                succes_iter_flag = False
            elif ro > c2:
                s_tmp = x - prev_x.flatten()
                y_tmp = grad - prev_grad.flatten()
                if (s_tmp @ y_tmp > eigval_eps * (s_tmp @ s_tmp)):
                    S[:, cur_iter] = s_tmp
                    Y[:, cur_iter] = y_tmp
                    cur_iter += 1
                    succes_iter_flag = True
                    prev_x = x
                    prev_grad = grad
                else:
                    succes_iter_flag = False
                mu *= sgm1
            else:
                s_tmp = x - prev_x.flatten()
                y_tmp = grad - prev_grad.flatten()
                if (s_tmp @ y_tmp > eigval_eps * (s_tmp @ s_tmp)):
                    S[:, cur_iter] = s_tmp
                    Y[:, cur_iter] = y_tmp
                    cur_iter += 1
                    succes_iter_flag = True
                    prev_x = x
                    prev_grad = grad
                else:
                    succes_iter_flag = False
    print(f"mu={mu}")
    return x_hst, prev_x, crit_list, func_diff_list