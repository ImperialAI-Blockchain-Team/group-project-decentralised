import web3 from './web3';
import { jobs} from "../abi/abi";

const address = '0x462C8cDc62Be544A2eEa3668eA9c13742E5DaEE8';

export default new web3.eth.Contract(jobs, address);