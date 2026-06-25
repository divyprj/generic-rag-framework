# Quantum Cryptography and Security

## The Quantum Threat to Modern Cryptography

Modern digital security relies on public-key (asymmetric) cryptography, including RSA, Elliptic Curve Cryptography (ECC), and Diffie-Hellman key exchanges. These protocols protect internet communications by exploiting mathematical problems that are intractable for classical computers, such as integer factorization and discrete logarithms. However, Shor's algorithm, published in 1994, solves these problems in polynomial time, enabling a sufficiently large quantum computer to compromise RSA and ECC.

A key concern is the Harvest-Now, Decrypt-Later (HNDL) attack. Adversaries are actively capturing and storing encrypted data today, anticipating the arrival of fault-tolerant quantum computers that can decrypt it in the future. To quantify this risk, Michele Mosca proposed a theorem stating that if the time required to migrate to quantum-resistant standards ($X$) plus the time that data must remain secure ($Y$) exceeds the time required to build a cryptographically relevant quantum computer ($Z$), then the security of the data is already compromised ($X + Y > Z$). For symmetric cryptography, such as AES-256, the threat is less severe: Grover's algorithm provides a quadratic speedup for database searches, effectively halving key security (reducing AES-256 to 128 bits of security), which can be mitigated by doubling key lengths.

## Post-Quantum Cryptography and the NIST Standardization Timeline

To mitigate the quantum threat, the National Institute of Standards and Technology (NIST) initiated a global standardization project in 2016 to identify algorithms resistant to both classical and quantum attacks. Post-Quantum Cryptography (PQC) relies on mathematical problems that are believed to be hard for both classical and quantum systems, such as lattice-based cryptography, multivariate equations, code-based cryptography, and hash-based signatures.

In August 2024, NIST reached a historic milestone by finalizing its first three official post-quantum cryptographic standards:
1. **FIPS 203 (ML-KEM):** The Module-Lattice-Based Key-Encapsulation Mechanism, derived from the CRYSTALS-Kyber algorithm. It is designated as the primary standard for general encryption, such as securing web traffic.
2. **FIPS 204 (ML-DSA):** The Module-Lattice-Based Digital Signature Algorithm, derived from the CRYSTALS-Dilithium algorithm. It is the primary standard for general-purpose digital signatures used in identity verification and document signing.
3. **FIPS 205 (SLH-DSA):** The Stateless Hash-Based Digital Signature Algorithm, derived from the SPHINCS+ algorithm. It serves as a backup signature standard, relying on different mathematical assumptions (secure hash functions) than lattice-based methods.

NIST continues to evaluate alternative algorithms, including FN-DSA (derived from Falcon), to ensure cryptographic diversity and provide fallback options should vulnerabilities be discovered in the primary standards.

## Quantum Key Distribution and the BB84 Protocol

While PQC relies on mathematical complexity, Quantum Key Distribution (QKD) secures communication using the laws of quantum physics. QKD allows two parties, Alice (the sender) and Bob (the receiver), to generate a shared, random secret key that can be used for symmetric encryption. 

The first and most famous QKD protocol is BB84, proposed in 1984 by Charles Bennett and Gilles Brassard. In BB84, Alice transmits single photons polarized in one of four states corresponding to two conjugate bases: the rectilinear basis ($0^\circ, 90^\circ$) and the diagonal basis ($45^\circ, 135^\circ$). Alice randomly selects a basis and a bit value for each photon. Bob measures the incoming photons, randomly choosing between the rectilinear and diagonal bases. After transmission, they perform "sifting" over a public classical channel, revealing only the bases they used. They discard measurements where their bases did not match, leaving a raw key.

Because of the no-cloning theorem, an eavesdropper, Eve, cannot copy Alice's photons. Any attempt by Eve to measure the photons alters their states, introducing a detectable error rate (quantum bit error rate, or QBER) in Bob's measurements. If the QBER exceeds a threshold (typically 11%), Alice and Bob discard the key, knowing the channel is compromised.

## Decoy States and Practical QKD Security

Implementing BB84 in the real world presents hardware challenges. True single-photon sources are difficult to manufacture, so practical QKD systems use weak coherent pulses (lasers) attenuated to contain less than one photon on average. However, some pulses inevitably contain two or more identical photons. This exposes the system to Photon Number Splitting (PNS) attacks, where Eve splits off the extra photon from a multi-photon pulse, measures it, and allows the remaining photon to pass to Bob without introducing errors.

To counter PNS attacks, the decoy state protocol was proposed by Hwang in 2003 and later refined by Lo, Ma, and Chen. In this approach, Alice randomly interleaves the signal pulses with "decoy" pulses of different intensities (typically empty, weak, and strong). Because Eve cannot distinguish a signal pulse from a decoy pulse during transmission, her PNS attack alters the transmission statistics of the decoy states differently than the signal states. By checking the detection rate and error rate of the decoy states, Bob can detect PNS attacks and securely calculate the secret key rate.
