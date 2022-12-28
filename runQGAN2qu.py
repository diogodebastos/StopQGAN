import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import uproot
from matplotlib import cm
from sklearn.preprocessing import minmax_scale
from qutip import *

import numpy as np
import pennylane as qml
qml.drawer.use_style('black_white')
import tensorflow as tf
np.set_printoptions(suppress=True)

dev = qml.device('cirq.simulator', wires=5)

def spherical_to_cartesian(theta,phi):
    x = np.sin(theta)*np.cos(phi)
    y = np.sin(theta)*np.sin(phi)
    z = np.cos(theta)
    return x,y,z

def cartesian_to_spherical(x,y,z):
    if x > 0:
        phi = np.arctan(y/x)
    elif x < 0 and y >= 0:
        phi = np.arctan(y/x) - np.pi
    elif x < 0 and y < 0:
        phi = np.arctan(y/x) + np.pi
    elif x == 0 and y > 0:
        phi = np.pi/2
    elif x == 0 and y < 0:
        phi = -np.pi/2
    elif x == 0 and y == 0:
        phi = 0 # Undefined

    theta = np.arccos(z)
    return theta, phi

def real(angles, **kwargs):
    qml.RY(angles[0], wires=0)
    qml.RZ(angles[1], wires=0)
    qml.RY(angles[2], wires=1)
    qml.RZ(angles[3], wires=1)

def real_rand(angles, **kwargs):
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0,1])
    qml.RX(angles[0], wires=0)
    qml.RY(angles[1], wires=0)
    qml.RZ(angles[2], wires=0)
    qml.RX(angles[3], wires=0)
    qml.RY(angles[4], wires=0)
    qml.RZ(angles[5], wires=0)
    qml.CNOT(wires=[0, 1])
    qml.RX(angles[3], wires=1)
    qml.RY(angles[4], wires=1)
    qml.RZ(angles[5], wires=1)
    qml.RX(angles[0], wires=1)
    qml.RY(angles[1], wires=1)
    qml.RZ(angles[2], wires=1)

def discriminator(w, **kwargs):
    qml.Hadamard(wires=0)
    qml.Hadamard(wires=1)
    qml.CNOT(wires=[0,1])
    qml.Barrier(wires=[0,4],only_visual=True)
    discriminator_layer(w[:8])
    discriminator_layer(w[8:16]) 
    discriminator_layer(w[16:24])
    qml.RX(w[24], wires=4)
    qml.RY(w[25], wires=4)
    qml.RZ(w[26], wires=4)
    
def discriminator_layer(w, **kwargs):
    qml.RX(w[0], wires=0)
    qml.RX(w[1], wires=1)
    qml.RX(w[2], wires=4)
    qml.RZ(w[3], wires=0)
    qml.RZ(w[4], wires=1)
    qml.RZ(w[5], wires=4)
    qml.MultiRZ(w[6], wires=[0, 1])
    qml.MultiRZ(w[7], wires=[1, 4])

def generator(w, **kwargs):
    qml.Hadamard(wires=0)
    qml.Hadamard(wires=1)
    qml.CNOT(wires=[0,1])
    qml.Barrier(wires=[0,4],only_visual=True)
    generator_layer(w[:11])
    generator_layer(w[11:22])
    generator_layer(w[22:33])
    qml.RX(w[33], wires=0)
    qml.RY(w[34], wires=0)
    qml.RZ(w[35], wires=0)
    qml.RX(w[36], wires=1)
    qml.RY(w[37], wires=1)
    qml.RZ(w[38], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RX(w[39], wires=0)
    qml.RY(w[40], wires=0)
    qml.RZ(w[41], wires=0)
    qml.RX(w[42], wires=1)
    qml.RY(w[43], wires=1)
    qml.RZ(w[44], wires=1)

def generator_layer(w):
    qml.RY(w[0], wires=0)
    qml.RY(w[1], wires=1)
    qml.RY(w[2], wires=2)
    qml.RY(w[3], wires=3)
    qml.RZ(w[4], wires=0)
    qml.RZ(w[5], wires=1)
    qml.RZ(w[6], wires=2)
    qml.RZ(w[7], wires=3)
    qml.MultiRZ(w[8], wires=[0, 1])
    qml.MultiRZ(w[9], wires=[2, 3])
    qml.MultiRZ(w[10], wires=[1, 2])

@qml.qnode(dev, interface='tf')
def real_disc_circuit(angles, disc_weights):
    real(angles)
    discriminator(disc_weights)
    return qml.expval(qml.PauliZ(4))

@qml.qnode(dev, interface='tf')
def gen_disc_circuit(gen_weights, disc_weights):
    generator(gen_weights)
    discriminator(disc_weights)
    return qml.expval(qml.PauliZ(4))   

def prob_real_true(disc_weights):
    true_disc_output = real_disc_circuit(angles, disc_weights)
    prob_real_true = (true_disc_output + 1) / 2.
    return prob_real_true


def prob_fake_true(gen_weights, disc_weights):
    fake_disc_output = gen_disc_circuit(gen_weights, disc_weights)
    prob_fake_true = (fake_disc_output + 1) / 2.
    return prob_fake_true

def disc_cost(disc_weights):
    cost = prob_fake_true(gen_weights, disc_weights) - prob_real_true(disc_weights)
    return cost

def gen_cost(gen_weights):
    return -prob_fake_true(gen_weights, disc_weights)

def train_disc(it,disc_loss):
    print("== Training discriminator")
    cost = lambda: disc_cost(disc_weights)

    for step in range(it+1):
        opt.minimize(cost, disc_weights)
        cost_val = cost().numpy()
        disc_loss += [cost_val]
        #if step % int((it/5)) == 0:    
        if step % 5 == 0:    
            print("Step {}: cost = {:.6f}".format(step, cost_val))
            print(" Prob. real as real: {:.3f} %".format(prob_real_true(disc_weights).numpy()*100))
            print(" Prob. fake as real: {:.3f} %".format(prob_fake_true(gen_weights, disc_weights).numpy()*100))

def train_gen(it,gen_loss):
    print("== Training generator")
    cost = lambda: gen_cost(gen_weights)

    for step in range(it+1):
        opt.minimize(cost, gen_weights)
        cost_val = cost().numpy()
        gen_loss += [cost_val]
        #if step % int((it/5)) == 0:    
        if step % 5 == 0:
            print("Step {}: Prob. fake as real = {:.3f} %".format(step, abs(cost_val*100)))

def invert_minmax(X_scaled,min,max,feature):
    X_std = (X_scaled - min) / (max-min)
    X=X_std*(feature.max()-feature.min())+feature.min()
    return X

def theta_to_lepPt(theta):
    return invert_minmax(theta,0,np.pi,sgLepPt)

def phi_to_lepChg(phi):
    return np.sign(invert_minmax(phi,0,np.pi,sgLepChg))

def theta_to_Met(theta):
    return invert_minmax(theta,0,np.pi,sgMet)

def phi_to_lepEta(phi):
    return invert_minmax(phi,0,2*np.pi,sgLepEta)

def infoQu(name,angles,bv_real,bv_fake):
    tlist = np.linspace(0, 1, 40)
    colors = cm.cool(tlist)

    theta,phi = angles
    x,y,z=spherical_to_cartesian(theta,phi)
    X,Y,Z = bv_real
    THETA,PHI=cartesian_to_spherical(X,Y,Z)
    Xf,Yf,Zf = bv_fake
    THETAf,PHIf=cartesian_to_spherical(Xf,Yf,Zf)
    
    bloch = Bloch()
    
    vec_og=[x,y,z]
    vec_real=[X,Y,Z]
    vec_fake=[Xf,Yf,Zf]

    #bloch.add_vectors(vec_og)
    bloch.add_vectors(vec_real)
    bloch.add_vectors(vec_fake)

    #bloch.vector_color = list(colors[0:40:19])
    bloch.vector_color = list(colors[0:40:39])

    print("==   "+ name +"   ==")
    print("og   | theta,phi:  {0:.3f}  {1:.3f}".format(theta,phi))
    print("real | THETA,PHI:  {0:.3f}  {1:.3f}".format(THETA,PHI))
    print("fake | THETA,PHI:  {0:.3f}  {1:.3f}".format(THETAf,PHIf))
    print("      ----------------------------")
    print("og   | x,y,z: {0:.3f} {1:.3f} {2:.3f}".format(x,y,z))
    print("real | X,Y,Z: {0:.3f} {1:.3f} {2:.3f}".format(X,Y,Z))
    print("fake | X,Y,Z: {0:.3f} {1:.3f} {2:.3f}".format(Xf,Yf,Zf))
    
    bloch.save("results/"+name+".pdf")

    return THETA, PHI, THETAf, PHIf, theta, phi

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description='Process the command line options')
    parser.add_argument('-e', '--event', type=int, required=True, help='Event')
    args = parser.parse_args()
    ev_idx = args.event
    sys.stdout = open("results/"+str(ev_idx)+".log", "w")
    print("Log for event: "+str(ev_idx))
    
    path = "/Users/ketchum/Desktop/STOP_nTuples/"
    tuples = "nTuples17_nanoAOD_v2021-10-15_test/"

    signal = "T2DegStop_550_520_bdt"
    treename="bdttree"

    branches = ["LepPt","LepChg","LepEta","Met","Jet1Pt","mt","HT", "NbLoose","Njet", "JetHBpt", "DrJetHBLep", "JetHBDeepCSV","XS","Nevt","Event","weight"]

    preSel = "(LepPt < 30) & (Met > 280) & (HT > 200) & (Jet1Pt > 110) & ((DPhiJet1Jet2 < 2.5) | (Jet2Pt < 60)) & (isTight == 1) & (Met < 1000) & (mt < 300)"

    # load root files
    sgTree = uproot.open(path + tuples + signal +".root:"+treename)
    # select important events
    sgDict = sgTree.arrays(branches,preSel,library="np")
    
    sgLepPt  = sgDict["LepPt"]
    sgLepChg = sgDict["LepChg"]
    sgLepEta = sgDict["LepEta"]
    sgMet = sgDict["Met"]

    lepPt_to_theta = minmax_scale(sgLepPt, feature_range=(0, np.pi))
    lepChg_to_phi = minmax_scale(sgLepChg, feature_range=(0, np.pi))
    lepEta_to_phi = minmax_scale(sgLepEta, feature_range=(0, 2*np.pi))
    Met_to_theta = minmax_scale(sgMet, feature_range=(0, np.pi))

    angles=[lepPt_to_theta[ev_idx],lepChg_to_phi[ev_idx],Met_to_theta[ev_idx],lepEta_to_phi[ev_idx]]

    x1,y1,z1=spherical_to_cartesian(lepPt_to_theta[ev_idx],lepChg_to_phi[ev_idx])
    x2,y2,z2=spherical_to_cartesian(Met_to_theta[ev_idx],lepEta_to_phi[ev_idx])

    np.random.seed(42)
    eps = 1e-2
    init_gen_weights = np.array([np.pi] + [0] * 44) + \
                    np.random.normal(scale=eps, size=(45,))
    init_disc_weights = np.random.normal(size=(27,))

    gen_weights = tf.Variable(init_gen_weights)
    disc_weights = tf.Variable(init_disc_weights)

    print("Probability classify real as real: {:.3f} %".format(prob_real_true(disc_weights).numpy()*100))
    print("Probability classify fake as real: {:.3f} %".format(prob_fake_true(gen_weights, disc_weights).numpy()*100))

    opt = tf.keras.optimizers.SGD(0.4)
    disc_loss = []
    gen_loss = []

    print("Cycle 1")
    train_disc(20,disc_loss)
    train_gen(5,gen_loss)
    print("\nCycle 2")
    train_disc(20,disc_loss)
    train_gen(5,gen_loss)
    print("\nCycle 3")
    train_disc(20,disc_loss)
    train_gen(5,gen_loss)

    print(" Prob. real as real: {:.3f} %".format(prob_real_true(disc_weights).numpy()*100))
    print(" Prob. fake as real: {:.3f} %".format(prob_fake_true(gen_weights, disc_weights).numpy()*100))

    obs = [qml.PauliX(0), qml.PauliY(0), qml.PauliZ(0),qml.PauliX(1), qml.PauliY(1), qml.PauliZ(1)]

    bloch_vector_real = qml.map(real, obs, dev, interface="tf")
    bloch_vector_fake = qml.map(generator, obs, dev, interface="tf")

    bv_real = bloch_vector_real(angles).numpy()
    bv_fake = bloch_vector_fake(gen_weights).numpy()
    difference = np.absolute(bv_real - bv_fake)

    x1,y1,z1,x2,y2,z2=bv_real
    X1,Y1,Z1,X2,Y2,Z2=bv_fake
    d1,d2,d3,d4,d5,d6=difference

    print("Real Bloch vector:      Q1=[{0:.3f}, {1:.3f}, {2:.3f}] \t Q2=[{3:.3f}, {4:.3f}, {5:.3f}]".format(x1,y1,z1,x2,y2,z2))
    print("Generator Bloch vector: Q1=[{0:.3f}, {1:.3f}, {2:.3f}] \t Q2=[{3:.3f}, {4:.3f}, {5:.3f}]".format(X1,Y1,Z1,X2,Y2,Z2))
    print("Difference:             Q1=[{0:.3f}, {1:.3f}, {2:.3f}] \t Q2=[{3:.3f}, {4:.3f}, {5:.3f}]".format(d1,d2,d3,d4,d5,d6))

    accuracy = difference / (np.pi)
    average_accuracy = 0
    for i in range(len(accuracy)):
        average_accuracy += accuracy[i]
    average_accuracy = average_accuracy / len(accuracy)
    average_accuracy = 1 - average_accuracy
    print("Accuracy: {0:.3f} %".format(average_accuracy*100))

    THETA1, PHI1, THETAf1, PHIf1, theta1, phi1 = infoQu(str(ev_idx)+"_Q1",angles[:2],bv_real[:3],bv_fake[:3])
    THETA2, PHI2, THETAf2, PHIf2, theta2, phi2 = infoQu(str(ev_idx)+"_Q2",angles[2:],bv_real[3:],bv_fake[3:])

    print("\n\nev: {0:.0f} | LepPt | LepChg | MET | LepEta".format(ev_idx))
    print("-------------------------------------")
    print("og     |  {0:.1f}  |   {1:.0f}    | {2:.0f} | {3:.2f}".format(theta_to_lepPt(theta1),phi_to_lepChg(phi1),theta_to_Met(theta2),phi_to_lepEta(phi2)))
    print("real   |  {0:.1f}  |   {1:.0f}    | {2:.0f} | {3:.2f}".format(theta_to_lepPt(THETA1),phi_to_lepChg(PHI1),theta_to_Met(THETA2),phi_to_lepEta(PHI2)))
    print("fake   | {0:.1f}   |   {1:.0f}    | {2:.0f} | {3:.2f}".format(theta_to_lepPt(THETAf1),phi_to_lepChg(PHIf1),theta_to_Met(THETAf2),phi_to_lepEta(PHIf2)))
    print("\t ")
    sys.stdout.close()