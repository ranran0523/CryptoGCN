﻿// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

using Microsoft.Research.SEAL.Tools;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Microsoft.Research.SEAL
{
    /// <summary>
    /// Generates matching secret key and public key.
    /// </summary>
    /// <remarks>
    /// Generates matching secret key and public key. An existing KeyGenerator can
    /// also at any time be used to generate relinearization keys and Galois keys.
    /// Constructing a KeyGenerator requires only a SEALContext.
    /// </remarks>
    public class KeyGenerator : NativeObject
    {
        /// <summary>
        /// Creates a KeyGenerator initialized with the specified SEALContext.
        /// </summary>
        /// <remarks>
        /// Creates a KeyGenerator initialized with the specified <see cref="SEALContext" />.
        /// Dynamically allocated member variables are allocated from the global memory pool.
        /// </remarks>
        /// <param name="context">The SEALContext</param>
        /// <exception cref="ArgumentException">if encryption parameters are not
        /// valid</exception>
        /// <exception cref="ArgumentNullException">if context is null</exception>
        public KeyGenerator(SEALContext context)
        {
            if (null == context)
                throw new ArgumentNullException(nameof(context));
            if (!context.ParametersSet)
                throw new ArgumentException("Encryption parameters are not set correctly");

            NativeMethods.KeyGenerator_Create(context.NativePtr, out IntPtr ptr);
            NativePtr = ptr;
        }

        /// <summary>
        /// Creates an KeyGenerator instance initialized with the specified
        /// SEALContext and specified previously secret key.
        /// </summary>
        /// <remarks>
        /// Creates an KeyGenerator instance initialized with the specified
        /// SEALContext and specified previously secret key. This can e.g. be used
        /// to increase the number of relinearization keys from what had earlier
        /// been generated, or to generate Galois keys in case they had not been
        /// generated earlier.
        /// </remarks>
        /// <param name="context">The SEALContext</param>
        /// <param name="secretKey">A previously generated secret key</param>
        /// <exception cref="ArgumentNullException">if either context or secretKey
        /// are null</exception>
        /// <exception cref="ArgumentException">if encryption parameters are not
        /// valid</exception>
        /// <exception cref="ArgumentException">if secretKey or publicKey is not
        /// valid for encryption parameters</exception>
        public KeyGenerator(SEALContext context, SecretKey secretKey)
        {
            if (null == context)
                throw new ArgumentNullException(nameof(context));
            if (null == secretKey)
                throw new ArgumentNullException(nameof(secretKey));
            if (!context.ParametersSet)
                throw new ArgumentException("Encryption parameters are not set correctly");
            if (!ValCheck.IsValidFor(secretKey, context))
                throw new ArgumentException("Secret key is not valid for encryption parameters");

            NativeMethods.KeyGenerator_Create(context.NativePtr,
                secretKey.NativePtr, out IntPtr ptr);
            NativePtr = ptr;
        }

        /// <summary>
        /// Returns a copy of the secret key.
        /// </summary>
        public SecretKey SecretKey
        {
            get
            {
                NativeMethods.KeyGenerator_SecretKey(NativePtr, out IntPtr secretKeyPtr);
                SecretKey secretKey = new SecretKey(secretKeyPtr);
                return secretKey;
            }
        }

        /// <summary>
        /// Generates a public key and stores the result in destination.
        /// </summary>
        /// <remarks>
        /// Generates a public key and stores the result in destination. Every time
        /// this function is called, a new public key will be generated.
        /// </remarks>
        /// <param name="destination">The public key to overwrite with the generated
        /// public key</param>
        public void CreatePublicKey(out PublicKey destination)
        {
            NativeMethods.KeyGenerator_CreatePublicKey(NativePtr, false, out IntPtr pubKeyPtr);
            destination = new PublicKey(pubKeyPtr);
        }

        /// <summary>
        /// Generates and returns a public key as a serializable object.
        /// </summary>
        /// <remarks>
        /// <para>
        /// Generates and returns a public key as a serializable object. Every time
        /// this function is called, a new public key will be generated.
        /// </para>
        /// <para>
        /// Half of the key data is pseudo-randomly generated from a seed to reduce
        /// the object size. The resulting serializable object cannot be used
        /// directly and is meant to be serialized for the size reduction to have an
        /// impact.
        /// </para>
        /// </remarks>
        public Serializable<PublicKey> CreatePublicKey()
        {
            NativeMethods.KeyGenerator_CreatePublicKey(NativePtr, true, out IntPtr pubKeyPtr);
            return new Serializable<PublicKey>(new PublicKey(pubKeyPtr));
        }

        /// <summary>
        /// Generates relinearization keys and stores the result in destination.
        /// </summary>
        /// <remarks>
        /// Generates relinearization keys and stores the result in destination.
        /// Every time this function is called, new relinearization keys will be
        /// generated.
        /// </remarks>
        /// <param name="destination">The relinearization keys to overwrite with
        /// the generated relinearization keys</param>
        /// <exception cref="InvalidOperationException">if the encryption
        /// parameters do not support keyswitching</exception>
        public void CreateRelinKeys(out RelinKeys destination)
        {
            if (!UsingKeyswitching())
                throw new InvalidOperationException("Encryption parameters do not support keyswitching");

            NativeMethods.KeyGenerator_CreateRelinKeys(NativePtr, false, out IntPtr relinKeysPtr);
            destination = new RelinKeys(relinKeysPtr);
        }

        /// <summary>
        /// Generates and returns relinearization keys as a serializable object.
        /// </summary>
        /// <remarks>
        /// <para>
        /// Generates and returns relinearization keys as a serializable object.
        /// Every time this function is called, new relinearization keys will be
        /// generated.
        /// </para>
        /// <para>
        /// Half of the key data is pseudo-randomly generated from a seed to reduce
        /// the object size. The resulting serializable object cannot be used
        /// directly and is meant to be serialized for the size reduction to have an
        /// impact.
        /// </para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">if the encryption
        /// parameters do not support keyswitching</exception>
        public Serializable<RelinKeys> CreateRelinKeys()
        {
            if (!UsingKeyswitching())
                throw new InvalidOperationException("Encryption parameters do not support keyswitching");

            NativeMethods.KeyGenerator_CreateRelinKeys(NativePtr, true, out IntPtr relinKeysPtr);
            return new Serializable<RelinKeys>(new RelinKeys(relinKeysPtr));
        }

        /// <summary>
        /// Generates Galois keys and stores the result in destination.
        /// </summary>
        /// <remarks>
        /// <para>
        /// Generates Galois keys and stores the result in destination. Every time
        /// this function is called, new Galois keys will be generated.
        /// </para>
        /// <para>
        /// This function creates specific Galois keys that can be used to apply
        /// specific Galois automorphisms on encrypted data. The user needs to give
        /// as input a vector of Galois elements corresponding to the keys that are
        /// to be created.
        /// </para>
        /// <para>
        /// The Galois elements are odd integers in the interval [1, M-1], where
        /// M = 2*N, and N = PolyModulusDegree. Used with batching, a Galois element
        /// 3^i % M corresponds to a cyclic row rotation i steps to the left, and
        /// a Galois element 3^(N/2-i) % M corresponds to a cyclic row rotation i
        /// steps to the right. The Galois element M-1 corresponds to a column rotation
        /// (row swap). In the polynomial view (not batching), a Galois automorphism by
        /// a Galois element p changes Enc(plain(x)) to Enc(plain(x^p)).
        /// </para>
        /// </remarks>
        /// <param name="galoisElts">The Galois elements for which to generate keys</param>
        /// <param name="destination">The Galois keys to overwrite with the generated
        /// Galois keys</param>
        /// <exception cref="InvalidOperationException">if the encryption
        /// parameters do not support keyswitching</exception>
        /// <exception cref="ArgumentException">if the Galois elements are not valid</exception>
        public void CreateGaloisKeys(IEnumerable<uint> galoisElts, out GaloisKeys destination)
        {
            if (null == galoisElts)
                throw new ArgumentNullException(nameof(galoisElts));
            if (!UsingKeyswitching())
                throw new InvalidOperationException("Encryption parameters do not support keyswitching");

            uint[] galoisEltsArr = galoisElts.ToArray();
            NativeMethods.KeyGenerator_CreateGaloisKeysFromElts(NativePtr,
                (ulong)galoisEltsArr.Length, galoisEltsArr, false, out IntPtr galoisKeysPtr);
            destination = new GaloisKeys(galoisKeysPtr);
        }

        /// <summary>
        /// Generates and returns Galois keys as a serializable object.
        /// </summary>
        /// <remarks>
        /// <para>
        /// Generates and returns Galois keys as a serializable object. Every time
        /// this function is called, new Galois keys will be generated.
        /// </para>
        /// <para>
        /// Half of the key data is pseudo-randomly generated from a seed to reduce
        /// the object size. The resulting serializable object cannot be used
        /// directly and is meant to be serialized for the size reduction to have an
        /// impact.
        /// </para>
        /// <para>
        /// This function creates specific Galois keys that can be used to apply
        /// specific Galois automorphisms on encrypted data. The user needs to give
        /// as input a vector of Galois elements corresponding to the keys that are
        /// to be created.
        /// </para>
        /// <para>
        /// The Galois elements are odd integers in the interval [1, M-1], where
        /// M = 2*N, and N = PolyModulusDegree. Used with batching, a Galois element
        /// 3^i % M corresponds to a cyclic row rotation i steps to the left, and
        /// a Galois element 3^(N/2-i) % M corresponds to a cyclic row rotation i
        /// steps to the right. The Galois element M-1 corresponds to a column rotation
        /// (row swap). In the polynomial view (not batching), a Galois automorphism by
        /// a Galois element p changes Enc(plain(x)) to Enc(plain(x^p)).
        /// </para>
        /// </remarks>
        /// <param name="galoisElts">The Galois elements for which to generate keys</param>
        /// <exception cref="ArgumentNullException">if galoisElts is null</exception>
        /// <exception cref="InvalidOperationException">if the encryption
        /// parameters do not support keyswitching</exception>
        /// <exception cref="ArgumentException">if the Galois elements are not valid</exception>
        public Serializable<GaloisKeys> CreateGaloisKeys(IEnumerable<uint> galoisElts)
        {
            if (null == galoisElts)
                throw new ArgumentNullException(nameof(galoisElts));
            if (!UsingKeyswitching())
                throw new InvalidOperationException("Encryption parameters do not support keyswitching");

            uint[] galoisEltsArr = galoisElts.ToArray();
            NativeMethods.KeyGenerator_CreateGaloisKeysFromElts(NativePtr,
                (ulong)galoisEltsArr.Length, galoisEltsArr, true, out IntPtr galoisKeysPtr);
            return new Serializable<GaloisKeys>(new GaloisKeys(galoisKeysPtr));
        }

        /// <summary>
        /// Generates Galois keys and stores the result in destination.
        /// </summary>
        /// <remarks>
        /// <para>
        /// Generates Galois keys and stores the result in destination. Every time
        /// this function is called, new Galois keys will be generated.
        /// </para>
        /// <para>
        /// The user needs to give as input a vector of desired Galois rotation step
        /// counts, where negative step counts correspond to rotations to the right
        /// and positive step counts correspond to rotations to the left. A step
        /// count of zero can be used to indicate a column rotation in the BFV scheme
        /// and complex conjugation in the CKKS scheme.
        /// </para>
        /// </remarks>
        /// <param name="steps">The rotation step counts for which to generate keys</param>
        /// <param name="destination">The Galois keys to overwrite with the generated
        /// Galois keys</param>
        /// <exception cref="ArgumentNullException">if steps is null</exception>
        /// <exception cref="InvalidOperationException">if the encryption parameters
        /// do not support batching and scheme is SchemeType.BFV</exception>
        /// <exception cref="InvalidOperationException">if the encryption
        /// parameters do not support keyswitching</exception>
        /// <exception cref="ArgumentException">if the step counts are not valid</exception>
        public void CreateGaloisKeys(IEnumerable<int> steps, out GaloisKeys destination)
        {
            if (null == steps)
                throw new ArgumentNullException(nameof(steps));
            if (!UsingKeyswitching())
                throw new InvalidOperationException("Encryption parameters do not support keyswitching");

            int[] stepsArr = steps.ToArray();
            NativeMethods.KeyGenerator_CreateGaloisKeysFromSteps(NativePtr,
                (ulong)stepsArr.Length, stepsArr, false, out IntPtr galoisKeysPtr);
            destination = new GaloisKeys(galoisKeysPtr);
        }

        /// <summary>
        /// Generates and returns Galois keys as a serializable object.
        /// </summary>
        /// <remarks>
        /// <para>
        /// Generates and returns Galois keys as a serializable object. Every time
        /// this function is called, new Galois keys will be generated.
        /// </para>
        /// <para>
        /// Half of the key data is pseudo-randomly generated from a seed to reduce
        /// the object size. The resulting serializable object cannot be used
        /// directly and is meant to be serialized for the size reduction to have an
        /// impact.
        /// </para>
        /// <para>
        /// The user needs to give as input a vector of desired Galois rotation step
        /// counts, where negative step counts correspond to rotations to the right
        /// and positive step counts correspond to rotations to the left. A step
        /// count of zero can be used to indicate a column rotation in the BFV scheme
        /// and complex conjugation in the CKKS scheme.
        /// </para>
        /// </remarks>
        /// <param name="steps">The rotation step counts for which to generate keys</param>
        /// <exception cref="ArgumentNullException">if steps is null</exception>
        /// <exception cref="InvalidOperationException">if the encryption parameters
        /// do not support batching and scheme is SchemeType.BFV</exception>
        /// <exception cref="InvalidOperationException">if the encryption
        /// parameters do not support keyswitching</exception>
        /// <exception cref="ArgumentException">if the step counts are not valid</exception>
        public Serializable<GaloisKeys> CreateGaloisKeys(IEnumerable<int> steps)
        {
            if (null == steps)
                throw new ArgumentNullException(nameof(steps));
            if (!UsingKeyswitching())
                throw new InvalidOperationException("Encryption parameters do not support keyswitching");

            int[] stepsArr = steps.ToArray();
            NativeMethods.KeyGenerator_CreateGaloisKeysFromSteps(NativePtr,
                (ulong)stepsArr.Length, stepsArr, true, out IntPtr galoisKeysPtr);
            return new Serializable<GaloisKeys>(new GaloisKeys(galoisKeysPtr));
        }

        /// <summary>
        /// Generates Galois keys and stores the result in destination.
        /// </summary>
        /// <remarks>
        /// <para>
        /// Generates Galois keys and stores the result in destination. Every time
        /// this function is called, new Galois keys will be generated.
        /// </para>
        /// <para>
        /// This function creates logarithmically many (in degree of the polynomial
        /// modulus) Galois keys that is sufficient to apply any Galois automorphism
        /// (e.g., rotations) on encrypted data. Most users will want to use this
        /// overload of the function.
        /// </para>
        /// <para>
        /// Precisely it generates 2*log(n)-1 number of Galois keys where n is the
        /// degree of the polynomial modulus. When used with batching, these keys
        /// support direct left and right rotations of power-of-2 steps of rows in BFV
        /// or vectors in CKKS and rotation of columns in BFV or conjugation in CKKS.
        /// </para>
        /// </remarks>
        /// <param name="destination">The Galois keys to overwrite with the generated
        /// Galois keys</param>
        /// <exception cref="InvalidOperationException">if the encryption
        /// parameters do not support keyswitching</exception>
        public void CreateGaloisKeys(out GaloisKeys destination)
        {
            if (!UsingKeyswitching())
                throw new InvalidOperationException("Encryption parameters do not support keyswitching");

            NativeMethods.KeyGenerator_CreateGaloisKeysAll(NativePtr, false, out IntPtr galoisKeysPtr);
            destination = new GaloisKeys(galoisKeysPtr);
        }

        /// <summary>
        /// Generates and returns Galois keys as a serializable object.
        /// </summary>
        /// <remarks>
        /// <para>
        /// Generates and returns Galois keys as a serializable object. Every time
        /// this function is called, new Galois keys will be generated.
        /// </para>
        /// <para>
        /// Half of the key data is pseudo-randomly generated from a seed to reduce
        /// the object size. The resulting serializable object cannot be used
        /// directly and is meant to be serialized for the size reduction to have an
        /// impact.
        /// </para>
        /// <para>
        /// This function creates logarithmically many (in degree of the polynomial
        /// modulus) Galois keys that is sufficient to apply any Galois automorphism
        /// (e.g., rotations) on encrypted data. Most users will want to use this
        /// overload of the function.
        /// </para>
        /// <para>
        /// Precisely it generates 2*log(n)-1 number of Galois keys where n is the
        /// degree of the polynomial modulus. When used with batching, these keys
        /// support direct left and right rotations of power-of-2 steps of rows in BFV
        /// or vectors in CKKS and rotation of columns in BFV or conjugation in CKKS.
        /// </para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">if the encryption
        /// parameters do not support keyswitching</exception>
        public Serializable<GaloisKeys> CreateGaloisKeys()
        {
            if (!UsingKeyswitching())
                throw new InvalidOperationException("Encryption parameters do not support keyswitching");

            NativeMethods.KeyGenerator_CreateGaloisKeysAll(NativePtr, true, out IntPtr galoisKeysPtr);
            return new Serializable<GaloisKeys>(new GaloisKeys(galoisKeysPtr));
        }

        /// <summary>
        /// Destroy native object.
        /// </summary>
        protected override void DestroyNativeObject()
        {
            NativeMethods.KeyGenerator_Destroy(NativePtr);
        }

        internal bool UsingKeyswitching()
        {
            NativeMethods.KeyGenerator_ContextUsingKeyswitching(NativePtr, out bool result);
            return result;
        }
    }
}
