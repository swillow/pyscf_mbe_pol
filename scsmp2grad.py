#!/usr/bin/env python
# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
(SCS)-MP2 analytical nuclear gradients
'''

import numpy
from pyscf import lib
from functools import reduce

from pyscf.ao2mo import _ao2mo
from pyscf.data.nist import BOHR

import scsmp2 

def kernel_rdm1 (mp, mo_coeff=None, mo_energy=None, verbose=0):
    from pyscf.grad.mp2 import _response_dm1, _index_frozen_active, _shell_prange

    mf_grad = mp._scf.nuc_grad_method()

    if mo_coeff is None: mo_coeff = mp.mo_coeff
    if mo_energy is None: mo_energy = mp._scf.mo_energy

    d1 = scsmp2._gamma1_intermediates(mp, mp.t2)
    doo, dvv = d1

# Set nocc, nvir for half-transformation of 2pdm.  Frozen orbitals are exculded.
# nocc, nvir should be updated to include the frozen orbitals when proceeding
# the 1-particle quantities later.
    mol = mp.mol
    with_frozen = not ((mp.frozen is None) or 
                       (isinstance(mp.frozen, (int, numpy.integer)) and mp.frozen == 0) )

    OA, VA, OF, VF = _index_frozen_active(mp.get_frozen_mask(), mp.mo_occ)
    orbo = mo_coeff[:,OA]
    orbv = mo_coeff[:,VA]
    nao, nocc = orbo.shape
    nvir = orbv.shape[1]

# Partially transform MP2 density matrix and hold it in memory
# The rest transformation are applied during the contraction to ERI integrals
    part_dm2 = _ao2mo.nr_e2(mp.t2.reshape(nocc**2,nvir**2),
                            numpy.asarray(orbv.T, order='F'), (0,nao,0,nao),
                            's1', 's1').reshape(nocc,nocc,nao,nao)
    part_dm2 = (part_dm2.transpose(0,2,3,1) * (mp.pt+mp.ps) -
                part_dm2.transpose(0,3,2,1) * mp.pt)*2.0

    hf_dm1 = mp._scf.make_rdm1(mo_coeff, mp.mo_occ)

    atmlst = range(mol.natm)
    offsetdic = mol.offset_nr_by_atom()
    diagidx = numpy.arange(nao)
    diagidx = diagidx*(diagidx+1)//2 + diagidx
    de = numpy.zeros((len(atmlst),3))
    Imat = numpy.zeros((nao,nao))
    fdm2 = lib.H5TmpFile()
    vhf1 = fdm2.create_dataset('vhf1', (len(atmlst),3,nao,nao), 'f8')

# 2e AO integrals dot 2pdm
    max_memory = max(0, mp.max_memory - lib.current_memory()[0])
    blksize = max(1, int(max_memory*.9e6/8/(nao**3*2.5)))

    for k, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = offsetdic[ia]
        ip1 = p0
        vhf = numpy.zeros((3,nao,nao))
        for b0, b1, nf in _shell_prange(mol, shl0, shl1, blksize):
            ip0, ip1 = ip1, ip1 + nf
            dm2buf = lib.einsum('pi,iqrj->pqrj', orbo[ip0:ip1], part_dm2)
            dm2buf+= lib.einsum('qi,iprj->pqrj', orbo, part_dm2[:,ip0:ip1])
            dm2buf = lib.einsum('pqrj,sj->pqrs', dm2buf, orbo)
            dm2buf = dm2buf + dm2buf.transpose(0,1,3,2)
            dm2buf = lib.pack_tril(dm2buf.reshape(-1,nao,nao)).reshape(nf,nao,-1)
            dm2buf[:,:,diagidx] *= .5

            shls_slice = (b0,b1,0,mol.nbas,0,mol.nbas,0,mol.nbas)
            eri0 = mol.intor('int2e', aosym='s2kl', shls_slice=shls_slice)
            Imat += lib.einsum('ipx,iqx->pq', eri0.reshape(nf,nao,-1), dm2buf)
            eri0 = None

            eri1 = mol.intor('int2e_ip1', comp=3, aosym='s2kl',
                             shls_slice=shls_slice).reshape(3,nf,nao,-1)
            de[k] -= numpy.einsum('xijk,ijk->x', eri1, dm2buf) * 2
            dm2buf = None
# HF part
            for i in range(3):
                eri1tmp = lib.unpack_tril(eri1[i].reshape(nf*nao,-1))
                eri1tmp = eri1tmp.reshape(nf,nao,nao,nao)
                vhf[i] += numpy.einsum('ijkl,ij->kl', eri1tmp, hf_dm1[ip0:ip1])
                vhf[i] -= numpy.einsum('ijkl,il->kj', eri1tmp, hf_dm1[ip0:ip1]) * .5
                vhf[i,ip0:ip1] += numpy.einsum('ijkl,kl->ij', eri1tmp, hf_dm1)
                vhf[i,ip0:ip1] -= numpy.einsum('ijkl,jk->il', eri1tmp, hf_dm1) * .5
            eri1 = eri1tmp = None
        vhf1[k] = vhf
        
# Recompute nocc, nvir to include the frozen orbitals and make contraction for
# the 1-particle quantities, see also the kernel function in ccsd_grad module.
    
    nao, nmo = mo_coeff.shape
    nocc = numpy.count_nonzero(mp.mo_occ > 0)
    Imat = reduce(numpy.dot, (mo_coeff.T, Imat, mp._scf.get_ovlp(), mo_coeff)) * -1

    dm1mo = numpy.zeros((nmo,nmo))
    if with_frozen:
        dco = Imat[OF[:,None],OA] / (mo_energy[OF,None] - mo_energy[OA])
        dfv = Imat[VF[:,None],VA] / (mo_energy[VF,None] - mo_energy[VA])
        dm1mo[OA[:,None],OA] = doo + doo.T
        dm1mo[OF[:,None],OA] = dco
        dm1mo[OA[:,None],OF] = dco.T
        dm1mo[VA[:,None],VA] = dvv + dvv.T
        dm1mo[VF[:,None],VA] = dfv
        dm1mo[VA[:,None],VF] = dfv.T
    else:
        dm1mo[:nocc,:nocc] = doo + doo.T
        dm1mo[nocc:,nocc:] = dvv + dvv.T

    dm1 = reduce(numpy.dot, (mo_coeff, dm1mo, mo_coeff.T))
    vhf = mp._scf.get_veff(mp.mol, dm1) * 2
    Xvo = reduce(numpy.dot, (mo_coeff[:,nocc:].T, vhf, mo_coeff[:,:nocc]))
    Xvo+= Imat[:nocc,nocc:].T - Imat[nocc:,:nocc]

    dm1mo += _response_dm1(mp, Xvo)

    Imat[nocc:,:nocc] = Imat[:nocc,nocc:].T
    im1 = reduce(numpy.dot, (mo_coeff, Imat, mo_coeff.T))

    hcore_deriv = mf_grad.hcore_generator(mol)
    s1 = mf_grad.get_ovlp(mol)
    zeta = lib.direct_sum('i+j->ij', mo_energy, mo_energy) * .5
    zeta[nocc:,:nocc] = mo_energy[:nocc]
    zeta[:nocc,nocc:] = mo_energy[:nocc].reshape(-1,1)
    zeta = reduce(numpy.dot, (mo_coeff, zeta*dm1mo, mo_coeff.T))

    dm1 = reduce(numpy.dot, (mo_coeff, dm1mo, mo_coeff.T))
    p1 = numpy.dot(mo_coeff[:,:nocc], mo_coeff[:,:nocc].T)
    vhf_s1occ = reduce(numpy.dot, (p1, mp._scf.get_veff(mol, dm1+dm1.T), p1))

    # Hartree-Fock part contribution
    dm1p = hf_dm1 + dm1*2
    dm1 += hf_dm1
    zeta += mf_grad.make_rdm1e(mo_energy, mo_coeff, mp.mo_occ)

    for k, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = offsetdic[ia]
# s[1] dot I, note matrix im1 is not hermitian
        de[k] += numpy.einsum('xij,ij->x', s1[:,p0:p1], im1[p0:p1])
        de[k] += numpy.einsum('xji,ij->x', s1[:,p0:p1], im1[:,p0:p1])
# h[1] \dot DM, contribute to f1
        h1ao = hcore_deriv(ia)
        de[k] += numpy.einsum('xij,ji->x', h1ao, dm1)
# -s[1]*e \dot DM,  contribute to f1
        de[k] -= numpy.einsum('xij,ij->x', s1[:,p0:p1], zeta[p0:p1]  )
        de[k] -= numpy.einsum('xji,ij->x', s1[:,p0:p1], zeta[:,p0:p1])
# -vhf[s_ij[1]],  contribute to f1, *2 for s1+s1.T
        de[k] -= numpy.einsum('xij,ij->x', s1[:,p0:p1], vhf_s1occ[p0:p1]) * 2
        de[k] -= numpy.einsum('xij,ij->x', vhf1[k], dm1p)

    de += mf_grad.grad_nuc(mol)
    return de, dm1


def mm_gradient (qm_mol, dm, mm_coords, mm_charges, l_nuc=True):
    # The interaction between QM atoms and MM particles
    # \sum_K d/dR (1/|r_K-R|) = \sum_K (r_K-R)/|r_K-R|^3

    if l_nuc:
        qm_coords = qm_mol.atom_coords()
        qm_charges = qm_mol.atom_charges()
        dr = qm_coords[:,None,:] - mm_coords
        r = numpy.linalg.norm(dr, axis=2)
        mm_grds = numpy.einsum('r,R,rRx,rR->Rx', qm_charges, mm_charges, dr, r**-3)
    else:
        mm_grds = numpy.zeros ( (mm_coords.shape), dtype=numpy.float64 )

    # The interaction between electron density and MM particles
    # d/dR <i| (1/|r-R|) |j> = <i| d/dR (1/|r-R|) |j> = <i| -d/dr (1/|r-R|) |j>
    #   = <d/dr i| (1/|r-R|) |j> + <i| (1/|r-R|) |d/dr j>
    for i, q in enumerate(mm_charges):
        with qm_mol.with_rinv_origin(mm_coords[i]):
            v = qm_mol.intor('int1e_iprinv')
        f =(numpy.einsum('ij,xji->x', dm, v) +
            numpy.einsum('ij,xij->x', dm, v.conj())) * -q
        mm_grds[i] += f

    # gradient = d/dR
    return mm_grds


def compute_ncore (mol):
    '''
    calculate the number of core
    mol : pyscf.Mole()
    '''

    ncore = 0
    znums = mol.atom_charges()
    for z in znums:
        if z <= 2: # H, He
            ncore += 0
        elif z <= 10: # Li ~ Ne
            ncore += 1
        elif z <= 18: # Na ~ Ar
            ncore += 5
        elif z <= 36: # K ~ Kr
            ncore += 9
        else:
            print ('nocre not supported')
            ncore += 18

    return ncore


def scsmp2_energy_gradient (mol,
                            l_mp2 = False,
                            mm_coords=None,
                            mm_charges=None,
                            l_pol=False):
    from pyscf import gto, scf, qmmm
    '''
    calculate the MP2 energy and gradients
    mm_coords : numpy.array ( (atom_xyz) )
    mm_charges : numpy.array ( (q) )
    '''

    mf = scf.RHF(mol)
    mf.chkfile = None
    mf = mf.run (verbose=0)

    ncore = 0 #compute_ncore (mol)
    
    rval = {}
    if mm_coords is None:
        mp = scsmp2.SCSMP2 (mf, l_mp2, frozen=ncore).run (verbose=0)
        qm_grad, rdm1 = kernel_rdm1 (mp)
        rval['ener'] = mp.e_tot
        rval['grad'] = qm_grad
    elif l_pol is False:
        mf_qmmm = qmmm.mm_charge (mf, mm_coords, mm_charges, unit='Bohr').run()
        mp_qmmm = scsmp2.SCSMP2 (mf_qmmm, l_mp2, frozen=ncore).run (verbose=0)
        qm_grad, rdm1 = kernel_rdm1 (mp_qmmm)
        mm_grad = mm_gradient (mol, rdm1, mm_coords, mm_charges)
        rval['ener'] = mp_qmmm.e_tot
        rval['grad'] = qm_grad
        rval['mm_grad'] = mm_grad
    else:
        mp = scsmp2.SCSMP2 (mf, l_mp2, frozen=ncore).run (verbose=0)
        qm_grad, rdm1 = kernel_rdm1 (mp)
        rval['ener'] = mp.e_tot
        rval['grad'] = qm_grad

        # Epol Energy at HF
        # using density = <Psi_I|Psi_I>
        # epol = <Psi_I:QI | H_I:QI | Psi_I:QI> - <Psi_I | H_I:QI | Psi_I>
        qm_dm = mf.make_rdm1()
        mf_qmmm = qmmm.mm_charge (mf, mm_coords, mm_charges, unit='Bohr').run()
        rval['epol'] = mf_qmmm.e_tot - mf_qmmm.energy_tot (qm_dm)
        
        mf_qmmm_grd = mf_qmmm.Gradients()
        grd_qm_Q = mf_qmmm_grd.kernel()
        grd_qm   = mf_qmmm_grd.kernel(mo_energy=mf.mo_energy, mo_coeff=mf.mo_coeff, mo_occ=mf.mo_occ)
        rval['epol_qm_grad'] = grd_qm_Q - grd_qm
        
        d_dm = mf_qmmm.make_rdm1() - qm_dm
        rval['epol_mm_grad'] = mm_gradient (mol, d_dm, mm_coords, mm_charges, l_nuc=False)
        
    return rval



def scsmp2_energy_density (mol,
                           l_mp2 = False,
                           mm_coords=None,
                           mm_charges=None):
    from pyscf import gto, scf, qmmm
    '''
    calculate the MP2 energy and gradients
    mm_coords : numpy.array ( (atom_xyz) )
    mm_charges : numpy.array ( (q) )
    '''

    mf = scf.RHF(mol)
    mf.chkfile = None
    mf = mf.run (verbose=0)

    ncore = 0 #compute_ncore (mol)
    
    rval = {}
    if mm_coords is None:
        mp = scsmp2.SCSMP2 (mf, l_mp2, frozen=ncore).run (verbose=0)
        qm_grad, rdm1 = kernel_rdm1 (mp)
        rval['ener'] = mp.e_tot
        rval['dm'] = rdm1
    else:
        mf_qmmm = qmmm.mm_charge (mf, mm_coords, mm_charges, unit='Bohr').run()
        mp_qmmm = scsmp2.SCSMP2 (mf_qmmm, l_mp2, frozen=ncore).run (verbose=0)
        qm_grad, rdm1 = kernel_rdm1 (mp_qmmm)
        rval['ener'] = mp_qmmm.e_tot
        rval['dm'] = rdm1

    return rval


if __name__ == '__main__':
    from pyscf import gto, scf

    ang2bohr = 1.0/BOHR
    atom_list = [
        [8 , numpy.array([0. , 0.     , 0.])*ang2bohr],
        [1 , numpy.array([0. , -0.757 , 0.587])*ang2bohr],
        [1 , numpy.array([0. , 0.757  , 0.587])*ang2bohr]]

    basis = 'cc-pvdz'
    print ("==QM: SCS-MP2==")
    #basis = 'aug-cc-pvdz'
    mol = gto.M(atom=atom_list, basis=basis, unit='Bohr', verbose=0)
    l_mp2 = False
    rval = scsmp2_energy_gradient (mol, l_mp2)
    print ('scsmp2_ener ', rval['ener'])
    print ('scsmp2_grad\n', rval['grad'])
    
    l_mp2 = True
    rval = scsmp2_energy_gradient (mol, l_mp2)
    print ('mp2_ener ', rval['ener'])
    print ('mp2_grad\n', rval['grad'])
    
    mm_coords = numpy.array ([
                           [ -0.5124238,  -1.3371964,   1.1037440],
                           [ -1.1027459,  -0.7016505,   1.5085072],
                           [  0.3518689,  -0.9291585,   1.1560423],
                           [  1.8155887,   0.1699667,   0.6965141],
                           [  1.9747686,  -0.1538473,  -0.1900739],
                           [  2.6801683,   0.1800438,   1.1071630],
                           [ -0.8629075,  -0.7370875,  -1.4967256],
                           [ -1.6770238,  -1.1121218,  -1.8325768],
                           [ -0.8050019,  -1.0566234,  -0.5962947],
                           [ -0.3552641,   1.7069145,  -0.1962745],
                           [ -0.6230563,   1.0286374,  -0.8163225],
                           [  0.4713443,   1.3880330,   0.1660307],
                           [  1.8677306,  -0.7082443,  -1.8727635],
                           [  2.0898068,  -0.3972344,  -2.7503665],
                           [  0.9159079,  -0.8085019,  -1.8873971] ])*ang2bohr


    print ("==QM/MM: SCS-MP2 + MM==")
    mm_charges = numpy.array ([-0.8, 0.4, 0.4,
                               -0.8, 0.4, 0.4,
                               -0.8, 0.4, 0.4,
                               -0.8, 0.4, 0.4,
                               -0.8, 0.4, 0.4])

    l_mp2 = False
    rval = scsmp2_energy_gradient (mol, l_mp2, mm_coords, mm_charges)
    print ('scsmp2_ener ', rval['ener'])
    print ('scsmp2_grad\n', rval['grad'])
    print ('mm_grad\n', rval['mm_grad'])
    
    l_mp2 = True
    rval = scsmp2_energy_gradient (mol, l_mp2, mm_coords, mm_charges)
    print ('mp2_ener ', rval['ener'])
    print ('mp2_grad\n', rval['grad'])
    print ('mm_grad\n', rval['mm_grad'])
    '''
    qm_mol = gto.M (atom=atom_list, basis=basis, unit='Bohr',verbose=0)
    mf = scf.RHF(qm_mol)
    mf.chkfile = None
    mf = mf.run (verbose=0)
    ncore = compute_ncore (qm_mol)
    mf_qmmm = qmmm.mm_charge (mf, mm_atom_list, mm_charges, unit='Bohr').run()
    mp_qmmm = mp2.MP2(mf_qmmm, frozen=ncore).run(verbose=0)
    qm_grad, rdm1 = kernel_rdm1 (mp_qmmm)
    mm_grad = mm_gradient (qm_mol, rdm1, mm_charges, mm_atom_list)
    print ('mp2_ener ', ener)
    print ('qm_grad\n', grad)
    print ('mm_grad\n', mm_grad)
    '''


