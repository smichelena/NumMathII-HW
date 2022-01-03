import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.sparse.linalg as splin
import scipy.sparse as sp

four_directions = [-1j, -1, 1, 1j] # s - w - e - n
eps = 1e-10

def bc(x, y):
    return 1 - abs(x)

def boundary(x, y, inside=True, strict=False):
    abxy = abs(x) + abs(y)
    if inside and strict:
        return abxy < 1-eps
    elif inside and not strict:
        return abxy <= 1+eps
    elif not inside and strict:
        return abxy > 1+eps
    else:
        return abxy >= 1-eps

def on_boundary(x, y):
    return abs(x) + abs(y) <= 1+eps and abs(x) + abs(y) >= 1-eps

def get_points_and_colors_in_Omega(N, strict=False):
    h = 1/(N+1)
    grid = np.empty((0,2))
    colors = []
    for j in range(-N-1,N+2):
        for i in range(-N-1,N+2):
            if boundary(h*i, h*j, strict=strict):
                grid = np.r_[grid, [[h*i, h*j]]]
                near_border = close_to_boundary(h*i, h*j, h)
                if near_border:
                    colors.append(1)
                if not near_border:
                    if on_boundary(h*i, h*j):
                        colors.append(2) # on the boundary
                    else:
                        colors.append(0) # not close or on boundary
    return grid, colors

def plot_points_in_Omega(N, strict=False):
    if not N % 2:
        raise Exception('N needs to be odd.')

    grid, colors = get_points_and_colors_in_Omega(N, strict=strict)
    plt.figure()
    # plt.grid()
    plt.scatter(grid[:,0], grid[:,1], c=colors)
    plt.show()

def get_stepsizes_and_bv(x, y, h):
    """
    Returns an array with the stepsizes in each
    direction: h_s = [h_s, h_w, h_e, h_n] and
    the boundary v = [v_s, v_w, v_e, h_n]
    """
    h_s = []
    v = []
    as_c = x + 1j*y
    for i,dir in enumerate(four_directions):
        new_c = as_c + dir*h
        added = False
        if boundary(new_c.real, new_c.imag, inside=False):
            if dir.imag:
                if dir.imag == 1:
                    y_new = 1 - abs(x)
                else:
                    y_new = abs(x) - 1
                h_s.append(abs(y_new - as_c.imag))
                v.append(bc(x, y + dir.imag*h_s[i]))
            else:
                if dir.real == 1:
                    x_new = 1 - abs(y)
                else:
                    x_new = abs(y) - 1
                h_s.append(abs(x_new - as_c.real))
                v.append(bc(x + dir.real*h_s[i], y))
            added = True
        if not added:
            h_s.append(h)
            v.append(0)
    return h_s, v

def diag_arrays(N):
    """
    Creates the diagonal arrays used in L_h
    """
    h = 1/(N+1)
    size = 0
    for i in range(N+1):
        if i == N:
            size += 2*i+1
        else:
            size += 2*(2*i+1)

    diags = []
    offsets = []
    for i in range(N+2):
        if i == 0:
            diag = -4/h**2*np.ones(size)
            diags.append(diag)
            offsets.append(0)
        elif i == 1:
            # diag = 1/h**2*np.ones(size)
            diag = np.ones(size)
            for j in range(N):
                diag[j+j*(j+1)] = 0
                diag[-(j+j*(j+1))-2] = 0
            diags.append(diag)
            diags.append(diag)
            offsets.append(1)
            offsets.append(-1)
        else:
            diag = np.zeros(size)
            ones_length = (i-1)*2-1
            to_ones_start = (i-2)**2
            diag[to_ones_start:ones_length+to_ones_start] = np.ones(ones_length)
            diag[-to_ones_start-ones_length-(i-1)*2:-to_ones_start-(i-1)*2] = np.ones(ones_length)
            # diag = [x/h**2 for x in diag]

            diags.append(diag)
            diags.append(diag)

            offsets.append(i*2-2)
            offsets.append(-(i*2-2))

    return diags, offsets

def create_Lh_from_diags(diags,offset):
    Lh = sp.diags(diags, offset, dtype=int)
    return sp.lil_matrix(Lh)

def get_bvp(N):
    h = 1/(N+1)

    Omega = get_points_and_colors_in_Omega(N, strict=True)[0]

    L_h = create_Lh_from_diags(*diag_arrays(N))
    f_h = []
    for i,(xy,row) in enumerate(zip(Omega,L_h)):
        x = xy[0]; y = xy[1]
        h_s, v = get_stepsizes_and_bv(x, y, h)
        sum_ = 0
        for h,dir in zip(h_s,v):
            sum_ -= dir/h**2
        f_h.append(sum_)

        # Divide by correct h
        for col in row.rows[0]:
            if col < i-1:
                L_h[i, col] /= h_s[0]**2
            elif col == i-1:
                L_h[i, col] /= h_s[1]**2
            elif col == i+1:
                L_h[i, col] /= h_s[2]**2
            elif col > i+1:
                L_h[i, col] /= h_s[3]**2
    L_h = sp.csr_matrix(L_h)

    return L_h, f_h

def solution(N):
    L_h, f_h = get_bvp(N)

    u_h = splin.spsolve(L_h, f_h)
    grid, _ = get_points_and_colors_in_Omega(N, strict=True)
    X = grid[:,0]
    Y = grid[:,1]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(X, Y, u_h)
    ax.title.set_text('Poisson equation diamond grid')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$z$')
    if os.path.isdir('../img'):
        plt.savefig('../img/poisson_eq_diamond.png')

def close_to_boundary(x, y, h):
    """
    Returns True if (x,y) is close to the boundary, else
    False
    """
    as_c = x + 1j*y
    for dir in four_directions:
        new_c = as_c + dir*h
        if on_boundary(x, y):
            return False
        if boundary(new_c.real, new_c.imag, inside=False):
            return True
    return False

def main():
    N = 45
    # get_stepsizes_and_bv(-0.5, -0.25, 1/(N+1))
    # get_bvp(N)
    # plot_points_in_Omega(N, strict=False)
    solution(N)
    if not os.path.isdir('../img'):
        plt.show()

if __name__ == '__main__':
    main()