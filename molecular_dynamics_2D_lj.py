#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulação de um gás ideal numa supercélula 2D usando dinâmica molecular.

Algoritmo de integração: velocity-verlet
Tipo de potencial: Lennard-Jones
Parâmetros de controle lidos do arquivo de imput do LAMMPS
Parâmetros de saída escritos no estilo XYZ, podendo ser usado com o Ovito

OBS: o script usa unidades reduzidas (units lj), ou seja, [m] = [sigma] = [epsilon] = k_B = 1
a conversão para unidades do SI é explicada em https://lammps.sandia.gov/doc/units.html

Autor: Alexandre Olivieri Kraus (olivieri.alexandre0@gmail.com)
Adaptado das demonstrações de Svaneborg em http://zqex.dk/index.php/method/lammps-demo
"""
import argparse
import pandas as pd
import numpy as np
import scipy.constants as const

def lj_aceleracao (epsilon, sigma, truncado, massa, distancia):
    aceleracao = np.zeros_like(distancia)
    for i in range(len(distancia)):
        if distancia[i] == 0:
            aceleracao[i] = 0.0
        elif distancia[i] < truncado:
            aceleracao[i] = (24 * epsilon * (2 * sigma**12/distancia[i] - sigma**6/distancia[i]))/massa
        else:
            aceleracao[i] = 0.0
    return aceleracao

def leitura_entrada_LAMMPS (filename_input):
    parametros = []
    with open(filename_input, "r") as arquivo:
        for linha in arquivo:
            # Gambiarra de case já que o python não tem esse controle de fluxo
            # Como se lê sequencialmente, a procura é feita na ordem que as variáveis aparecem
            if linha.startswith("variable npart"):
                #particulas_total
                tmp = linha.split()
                parametros.append(int(tmp[-1]))
            
            if linha.startswith("region"):
                # parametro_rede_x e parametro_rede_y
                tmp = linha.split()
                parametros.append(float(tmp[4]) - float(tmp[3]))
                parametros.append(float(tmp[6]) - float(tmp[5]))

            if linha.startswith("mass"):
                # particulas_massa
                parametros.append(float(linha.split()[-1]))
            
            if linha.startswith("pair_coeff"):
                # lj_epsilon, lj_sigma e lj_truncado
                tmp = linha.split()
                parametros.append(float(tmp[-3]))
                parametros.append(float(tmp[-2]))
                parametros.append(float(tmp[-1]))

            if linha.startswith("velocity"):
                # velocidade_inicial
                tmp = linha.split()
                parametros.append(float(tmp[3]))

            if linha.startswith("timestep"):
                # passo_tamanho
                parametros.append(float(linha.split()[-1]))
            
            if linha.startswith("run"):
                # passo_total
                parametros.append(int(linha.split()[-1]))
    
    return parametros

def escrita_saida_LAMMPS (filename_output, parametro_rede_x, parametro_rede_y, posicao, velocidade, total_passos):
    with open(filename_output, "w") as output:
        for passo in range(total_passos):
            output.write("ITEM: TIMESTEP\n{:}\n".format(passo))
            output.write("ITEM: NUMBER OF ATOMS\n{:}\n".format(len(posicao)))
            output.write("ITEM: BOX BOUNDS pp pp pp\n0.0 {:}\n0.0 {:}\n0.0 0.0\n".format(parametro_rede_x, parametro_rede_y))
            output.write("ITEM: ATOMS id type x y z vx vy vz\n")
            for particula in range(len(posicao)):
                output.write(
                    "{:} 1 {:} {:} 0.0 {:} {:} 0.0\n".format(
                        particula,
                        posicao[particula, 0, passo],
                        posicao[particula, 1, passo],
                        velocidade[particula, 0, passo],
                        velocidade[particula, 1, passo]
                    )
                )

parser = argparse.ArgumentParser(
    description="Simulação de um gás ideal numa caixa 2D usando dinâmica molecular com integrador de velocity-verlet e potencial tipo Lennard-Jones."
)
# Define os parâmetros posicionais obrigatórios
parser.add_argument("input", type = str)
parser.add_argument("output", type = str)
# Lê os parâmetros posicionais
parametros = parser.parse_args()

[
    particulas_total,
    parametro_rede_x,
    parametro_rede_y,
    particulas_massa,
    lj_epsilon,
    lj_sigma,
    lj_truncado,
    velocidade_inicial,
    passo_tamanho,
    passo_total
] = leitura_entrada_LAMMPS(parametros.input)

# População da supercélula
particulas_posicao = np.zeros((particulas_total, 2, passo_total))
particulas_posicao[:, :, 0] = np.dot(np.random.random_sample((particulas_total, 2)), np.array([[parametro_rede_x, 0.0], [0.0, parametro_rede_y]]))

# Velocidades iniciais
particulas_velocidade = np.zeros_like(particulas_posicao)
# Valores aleatorios dentro do intervalo [-velocidade_inicial, velocidade_inicial) em cada eixo
particulas_velocidade[:, :, 0] = 2 * velocidade_inicial * np.random.random_sample((particulas_total, 2)) - velocidade_inicial

particulas_delta_posicao = np.zeros((particulas_total, particulas_total, 2, passo_total))
# Acelerações iniciais
for i in range(particulas_total):
    for j in range(particulas_total):
        # Cálculo das distâncias entre partículas
        particulas_delta_posicao[i, j, :, 0] = particulas_posicao[j, :, 0] - particulas_posicao[i, :, 0]
        # Se essa distância é maior que a metade do parâmetro de rede, quer dizer que a partícula está mais próxima da imagem do par
        if np.absolute(particulas_delta_posicao[i, j, 0, 0]) > 0.5*parametro_rede_x:
            posicao_imagem = particulas_posicao[j, 0, 0] - parametro_rede_x
            particulas_delta_posicao[i, j, 0, 0] = posicao_imagem - particulas_posicao[i, 0, 0]
        if np.absolute(particulas_delta_posicao[i, j, 1, 0]) > 0.5*parametro_rede_y:
            posicao_imagem = particulas_posicao[j, 1, 0] - parametro_rede_y
            particulas_delta_posicao[i, j, 1, 0] = posicao_imagem - particulas_posicao[i, 1, 0]

particulas_aceleracao = np.zeros_like(particulas_posicao)
for i in range(particulas_total):
    particulas_aceleracao[i, 0, 0] = np.sum(lj_aceleracao(lj_epsilon, lj_sigma, lj_truncado, particulas_massa, particulas_delta_posicao[i, :, 0, 0]))# resultante em x
    particulas_aceleracao[i, 1, 0] = np.sum(lj_aceleracao(lj_epsilon, lj_sigma, lj_truncado, particulas_massa, particulas_delta_posicao[i, :, 1, 0]))# resultante em y

# Algoritmo velocity-verlet para o resto dos (passo_total - 1) passos
for passo in range(1, passo_total):
        # Atualização das posições
        particulas_posicao[:, :, passo] = particulas_posicao[:, :, passo - 1] + particulas_velocidade[:, :, passo - 1] * passo_tamanho + 0.5 * particulas_aceleracao[:, :, passo - 1] * passo_tamanho**2

        # Satisfação das condições periódicas de contorno
        for i in range(particulas_total):
            if particulas_posicao[i, 0, passo] >  parametro_rede_x:
                particulas_posicao[i, 0, passo] = particulas_posicao[i, 0, passo] % parametro_rede_x
            elif particulas_posicao[i, 0, passo] < 0:
                particulas_posicao[i, 0, passo] = parametro_rede_x - particulas_posicao[i, 0, passo] % parametro_rede_x
                
            if particulas_posicao[i, 1, passo] >  parametro_rede_y:
                particulas_posicao[i, 1, passo] = particulas_posicao[i, 1, passo] % parametro_rede_y
            elif particulas_posicao[i, 1, passo] < 0:
                particulas_posicao[i, 1, passo] = parametro_rede_y - particulas_posicao[i, 1, passo] % parametro_rede_y
        
        # Atualização das distâncias
        for i in range(particulas_total):
            for j in range(particulas_total):
                particulas_delta_posicao[i, j, :, passo] = particulas_posicao[j, :, passo] - particulas_posicao[i, :, passo]
                if np.absolute(particulas_delta_posicao[i, j, 0 , passo]) > 0.5*parametro_rede_x:
                    posicao_imagem = particulas_posicao[j, 0, passo] - parametro_rede_x
                    particulas_delta_posicao[i, j, 0, passo] = posicao_imagem - particulas_posicao[i, 0, passo]
                if np.absolute(particulas_delta_posicao[i, j, 1 , passo]) > 0.5*parametro_rede_y:
                    posicao_imagem = particulas_posicao[j, 1, passo] - parametro_rede_y
                    particulas_delta_posicao[i, j, 1, passo] = posicao_imagem - particulas_posicao[i, 1, passo]
        
            # Atualização das acelerações
            particulas_aceleracao[i, 0, passo] = np.sum(lj_aceleracao(lj_epsilon, lj_sigma, lj_truncado, particulas_massa, particulas_delta_posicao[i, :, 0, passo]))# resultante em x
            particulas_aceleracao[i, 1, passo] = np.sum(lj_aceleracao(lj_epsilon, lj_sigma, lj_truncado, particulas_massa, particulas_delta_posicao[i, :, 1, passo]))# resultante em y
        
        # Atualização das velocidades
        particulas_velocidade[:, :, passo] = particulas_velocidade[:, :, passo - 1] + 0.5 * (particulas_aceleracao[:, :, passo - 1] + particulas_aceleracao[:, :, passo]) * passo_tamanho

escrita_saida_LAMMPS(
    parametros.output,
    parametro_rede_x,
    parametro_rede_y,
    particulas_posicao,
    particulas_velocidade,
    passo_total
)
