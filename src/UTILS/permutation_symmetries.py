import numpy as np


def check_permutation_symmetries_complex_orbitals(one_body_tensor, two_body_tensor):
    # Works for both spin-orbital and orbital tensors
    symm_check_passed = True
    num_orbitals = one_body_tensor.shape[0]
    for p in range(num_orbitals):
        for q in range(num_orbitals):
            if not np.allclose(one_body_tensor[p, q], np.conj(one_body_tensor[q, p])):
                symm_check_passed = False
                print(
                    f"one_body_tensor[{p}, {q}] != np.conj(one_body_tensor[{q}, {p}]): {one_body_tensor[p, q]} != {np.conj(one_body_tensor[q, p])}"
                )

            for r in range(num_orbitals):
                for s in range(num_orbitals):
                    if (
                        not np.allclose(
                            two_body_tensor[p, q, r, s], two_body_tensor[r, s, p, q]
                        )
                        or not np.allclose(
                            two_body_tensor[p, q, r, s],
                            np.conj(two_body_tensor[q, p, s, r]),
                        )
                        or not np.allclose(
                            two_body_tensor[p, q, r, s],
                            np.conj(two_body_tensor[s, r, q, p]),
                        )
                    ):
                        symm_check_passed = False

                        print(
                            f"Permutation check of two body tensor failed.\n"
                            + f"two_body_tensor[{p},{q},{r},{s}] = {two_body_tensor[p,q,r,s]}\n"
                            + f"two_body_tensor[{r},{s},{p},{q}] = {two_body_tensor[r,s,p,q]}\n"
                            # + f"two_body_tensor[{q},{p},{r},{s}] = {two_body_tensor[q,p,r,s]}\n"
                            # + f"two_body_tensor[{p},{q},{s},{r}] = {two_body_tensor[p,q,s,r]}\n"
                            + f"two_body_tensor[{q},{p},{s},{r}] = {two_body_tensor[q,p,s,r]}\n"
                            # + f"two_body_tensor[{r},{s},{q},{p}] = {two_body_tensor[r,s,q,p]}\n"
                            # + f"two_body_tensor[{s},{r},{p},{q}] = {two_body_tensor[s,r,p,q]}\n"
                            + f"two_body_tensor[{s},{r},{q},{p}] = {two_body_tensor[s,r,q,p]}\n"
                        )
    return symm_check_passed


def check_permutation_symmetries_real_orbitals(one_body_tensor, two_body_tensor):
    # Works for both spin-orbital and orbital tensors
    # The symmetries tested here are valid only if the underlying ORBITALS (or SPIN-ORBITALS) are real,
    # NOT if the one_body_tensor and two_body_tensor elements are real.
    # The orbitals can be complex, while the elements of the one_body_tensor and two_body_tensor are still real.

    symm_check_passed = True
    num_orbitals = one_body_tensor.shape[0]
    for p in range(num_orbitals):
        for q in range(num_orbitals):
            if not np.allclose(one_body_tensor[p, q], one_body_tensor[q, p]):
                symm_check_passed = False
                print(
                    f"one_body_tensor[{p}, {q}] != one_body_tensor[{q}, {p}]: {one_body_tensor[p, q]} != {one_body_tensor[q, p]}"
                )

            for r in range(num_orbitals):
                for s in range(num_orbitals):
                    if (
                        not np.allclose(
                            two_body_tensor[p, q, r, s], two_body_tensor[r, s, p, q]
                        )
                        or not np.allclose(
                            two_body_tensor[p, q, r, s], two_body_tensor[q, p, r, s]
                        )
                        or not np.allclose(
                            two_body_tensor[p, q, r, s], two_body_tensor[p, q, s, r]
                        )
                        or not np.allclose(
                            two_body_tensor[p, q, r, s], two_body_tensor[q, p, s, r]
                        )
                        or not np.allclose(
                            two_body_tensor[p, q, r, s], two_body_tensor[r, s, q, p]
                        )
                        or not np.allclose(
                            two_body_tensor[p, q, r, s], two_body_tensor[s, r, p, q]
                        )
                        or not np.allclose(
                            two_body_tensor[p, q, r, s], two_body_tensor[s, r, q, p]
                        )
                    ):
                        symm_check_passed = False

                        print(
                            f"Permutation check of two body tensor failed.\n"
                            + f"two_body_tensor[{p},{q},{r},{s}] = {two_body_tensor[p,q,r,s]}\n"
                            + f"two_body_tensor[{r},{s},{p},{q}] = {two_body_tensor[r,s,p,q]}\n"
                            + f"two_body_tensor[{q},{p},{r},{s}] = {two_body_tensor[q,p,r,s]}\n"
                            + f"two_body_tensor[{p},{q},{s},{r}] = {two_body_tensor[p,q,s,r]}\n"
                            + f"two_body_tensor[{q},{p},{s},{r}] = {two_body_tensor[q,p,s,r]}\n"
                            + f"two_body_tensor[{r},{s},{q},{p}] = {two_body_tensor[r,s,q,p]}\n"
                            + f"two_body_tensor[{s},{r},{p},{q}] = {two_body_tensor[s,r,p,q]}\n"
                            + f"two_body_tensor[{s},{r},{q},{p}] = {two_body_tensor[s,r,q,p]}\n"
                        )
    return symm_check_passed
