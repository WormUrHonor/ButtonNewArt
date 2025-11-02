using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using System;
using TMPro;
using UnityEngine.Rendering;
using System.Linq.Expressions;
using System.Runtime.CompilerServices;
public class Recorder : MonoBehaviour
{
    public GameObject player1;
    public GameObject player2;

    public HPBar player1HPBar;
    public HPBar player2HPBar;

    // public neuralNetworkAI neuralNetworkAI;

    private string filename;
    private string filePath;
    private string directoryPath;

    private string currentCSVLine;

    // Start is called before the first frame update
    void Start()
    {
        filename = DateTime.Now.ToString("yyyyMMddHHmmss") + ".csv";
        directoryPath = Application.dataPath.Substring(0, Application.dataPath.LastIndexOf('/')) + "\\Recording\\";
        filePath = directoryPath + filename;

        System.IO.Directory.CreateDirectory(directoryPath);

        Action<string, StreamWriter> WritePlayerStateHeader = (playerPrefix, sw) =>
        {
            sw.Write(playerPrefix + "HP;");
            sw.Write(playerPrefix + "XCoord;");
            sw.Write(playerPrefix + "YCoord;");
            sw.Write(playerPrefix + "LeftPressed;");
            sw.Write(playerPrefix + "RightPressed;");
            sw.Write(playerPrefix + "JumpPressed;");
            sw.Write(playerPrefix + "DashPressed;");
            sw.Write(playerPrefix + "AttackPressed;");
            sw.Write(playerPrefix + "DownPressed;");
            sw.Write(playerPrefix + "IsGrounded;");
            sw.Write(playerPrefix + "IsDashing;");
            sw.Write(playerPrefix + "IsAttacking;");
            sw.Write(playerPrefix + "IsHit;");
            sw.Write(playerPrefix + "IsInHitStun;");
            sw.Write(playerPrefix + "IsLocked;");
        };

        using (StreamWriter sw = new StreamWriter(filePath, true))
        {
            WritePlayerStateHeader("P1", sw);
            WritePlayerStateHeader("P2", sw);

            sw.Write("Distance"); //Last column has no semicolon
            sw.Write("\n");
        }
    }

    private void Update()
    {
        currentCSVLine = "";

        RecordPlayerState(player1, player1HPBar);
        RecordPlayerState(player2, player2HPBar);

        using (StreamWriter sw = new StreamWriter(filePath, true))
        {
            AddItemToCSV((player1.transform.position - player2.transform.position).magnitude.ToString(), sw, false); //Last column has no semicolon
            AddItemToCSV("\n", sw, false);
        }

        //neuralNetworkAI.Submit(currentCSVLine);
    }

    public void RecordPlayerState(GameObject player, HPBar playerHPBar)
    {
        Movement playerMovement = player.GetComponent<Movement>();

        using (StreamWriter sw = new StreamWriter(filePath, true))
        {
            AddItemToCSV(playerHPBar.health.currentHP.ToString(), sw);
            AddItemToCSV(player.transform.position.x.ToString(), sw);
            AddItemToCSV(player.transform.position.y.ToString(), sw);
            AddItemToCSV(playerMovement.leftPressed.ToString(), sw);
            AddItemToCSV(playerMovement.rightPressed.ToString(), sw);
            AddItemToCSV(playerMovement.jumpPressed.ToString(), sw);
            AddItemToCSV(playerMovement.dashPressed.ToString(), sw);
            AddItemToCSV(playerMovement.attackPressed.ToString(), sw);
            AddItemToCSV(playerMovement.downPressed.ToString(), sw);
            AddItemToCSV(playerMovement.isGrounded.ToString(), sw);
            AddItemToCSV(playerMovement.dashing.ToString(), sw);
            AddItemToCSV(playerMovement.attacking.ToString(), sw);
            AddItemToCSV(playerMovement.isHit.ToString(), sw);
            AddItemToCSV(playerMovement.inHitstun.ToString(), sw);
            AddItemToCSV(playerMovement.locked.ToString(), sw);
        }
    }


    private void AddItemToCSV(string item, StreamWriter sw, bool addSemiColon = true)
    {
        string suffix = addSemiColon ? ";" : "";
        string lineItem = item + suffix;

        sw.Write(lineItem);
        currentCSVLine += lineItem;
    }
}
